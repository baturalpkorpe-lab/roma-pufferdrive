"""patch_drive_stopsigns.py -- put STOP SIGNS into the agent's observation.

    python patch_drive_stopsigns.py --drive_h /scratch/$USER/PufferDrive/pufferlib/ocean/drive/drive.h
    python patch_drive_stopsigns.py --drive_h ... --revert

WHY
Stop signs are type 7. init_grid_map gates on `type > 3 && type < 7`, so signs
never enter the grid map and never reach the policy -- verified in drive.h, and
the comment says so: "Only Road Edges, Lines, and Lanes in grid map".

Measured consequence: at alpha=0 the policy's junction minTTC is 0.48 of the
human value at all-way stops (>=3 signs), 0.58 at priority junctions (1-2), and
0.63 at signalised ones (no signs). The more signs a junction has, the worse it
does relative to a human -- which is what blindness to the sign looks like. And
speed_zone sits at 7.7 against a human 5.1 while barely responding to the role,
so it is a perception limit rather than a driving style.

THREE CHANGES, AND THE SECOND IS THE ONE THAT MATTERS

1. The type gates (three of them: two map-bounds passes and the grid
   insertion). `type < 7` -> `type < 8`. Signs only; crosswalks (8), speed
   bumps (9) and driveways (10) stay out. Crosswalks mark signalised junctions
   whose light state does not exist anywhere in the env, so they would be
   decoration, and all four would compete for the fixed 128 observation slots.

2. The insertion loops iterate SEGMENTS: `j < array_size - 1`. Every stop sign
   in the dataset is a single point (checked across five maps: type 7 is always
   array_size 1), so that loop body never executes and a sign inserts NOTHING.
   Widening the gate on its own is a silent no-op. Both the counting pass and
   the populate pass are rewritten to emit one entry for a single-point entity,
   and they must stay identical or the allocation and the fill disagree.

3. compute_observations reads traj_x[geometry_idx + 1] after only checking
   geometry_idx < array_size, so a single-point entity reads one past the end.
   Note this is ALREADY reachable without any of the above: map_002.bin has a
   type-4 lane with array_size 1. Collapse those to a zero-length segment.

WHAT DOES NOT CHANGE
obs_dim. The road observation is a fixed 128 slots x 7 features and the 7th is
already the type (`obs[idx+6] = entity->type - 4.0f`, giving lane 0, line 1,
edge 2). A sign arrives as 3.0 in a feature that already exists, so existing
checkpoints still load and this is a fine-tune rather than a retrain.

The cost of that: the type feature is ORDINAL, so a policy that has only seen
0, 1, 2 will read a sign as "more edge-like than an edge" until it learns
otherwise. ROAD_FEATURES_ONEHOT 13 exists unused and would avoid this, at the
price of changing obs_dim and forcing a from-scratch run. Try the cheap version
first; the expensive one is only justified if signs get treated as walls.
"""

import argparse
import shutil
import sys

GATE_OLD = "env->entities[i].type < 7"
GATE_NEW = "env->entities[i].type < 8"

LOOP_OLD = """            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;"""

LOOP_NEW = """            // A single-point entity (every stop sign, and the odd degenerate
            // lane) has no j+1, so the original `j < array_size - 1` never
            // ran its body and the entity was dropped silently. Emit one
            // entry at its own position instead. The counting pass and the
            // populate pass must agree exactly or the allocation is wrong.
            int nseg = (env->entities[i].array_size > 1) ? env->entities[i].array_size - 1 : 1;
            for (int j = 0; j < nseg; j++) {
                int j2 = (j + 1 < env->entities[i].array_size) ? j + 1 : j;
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j2]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j2]) / 2;"""

OBS_OLD = """            float end_x = entity->traj_x[geometry_idx + 1];
            float end_y = entity->traj_y[geometry_idx + 1];"""

OBS_NEW = """            // geometry_idx was validated against array_size, but this reads
            // ONE PAST it. Reachable today without stop signs: map_002.bin
            // carries a type-4 lane with array_size 1. Collapse single-point
            // entities to a zero-length segment; the hypot > 0 guard below
            // then leaves cos/sin at 0, and the non-zero relative position
            // keeps it distinguishable from a zero-padded empty slot.
            int gi2 = (geometry_idx + 1 < entity->array_size) ? geometry_idx + 1 : geometry_idx;
            float end_x = entity->traj_x[gi2];
            float end_y = entity->traj_y[gi2];"""

EDITS = [("type gate", GATE_OLD, GATE_NEW, 3),
         ("segment loop", LOOP_OLD, LOOP_NEW, 2),
         ("observation read", OBS_OLD, OBS_NEW, 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drive_h", required=True)
    p.add_argument("--revert", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    # newline="" both ways: without it Python rewrites every line ending to
    # the host convention, so a patch run from a Windows checkout would
    # silently convert the whole file to CRLF.
    src = open(a.drive_h, encoding="utf-8", newline="").read()
    edits = [(n, b, x, c) for n, x, b, c in EDITS] if a.revert else EDITS

    # Check every edit BEFORE writing anything: a half-applied patch to the
    # counting pass but not the populate pass corrupts the grid silently.
    plan = []
    for name, old, new, want in edits:
        got = src.count(old)
        if got == 0 and src.count(new) >= want:
            print("  %-18s already %s" % (name, "reverted" if a.revert else "applied"))
            continue
        if got != want:
            print("  %-18s FAIL: found %d occurrences, expected %d" % (name, got, want))
            print("     drive.h does not match what this patch was written "
                  "against. Do not force it -- re-read the file and update "
                  "the patch.")
            return 1
        plan.append((name, old, new, got))

    if not plan:
        print("\n  nothing to do")
        return 0
    for name, _, _, got in plan:
        print("  %-18s %d occurrence(s) -> will patch" % (name, got))
    if a.dry_run:
        print("\n  dry run, nothing written")
        return 0

    bak = a.drive_h + (".pre_signs" if not a.revert else ".pre_revert")
    shutil.copy2(a.drive_h, bak)
    for _, old, new, _ in plan:
        src = src.replace(old, new)
    open(a.drive_h, "w", encoding="utf-8", newline="").write(src)
    print("\n  wrote %s   (backup: %s)" % (a.drive_h, bak))
    print("\n  NOW REBUILD THE BINDING, then verify signs actually arrive:")
    print("    obs_dim is IDENTICAL either way, so nothing downstream will")
    print("    notice if the rebuild did not happen. The road type feature")
    print("    must now show 3.0 as well as 0.0/1.0/2.0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
