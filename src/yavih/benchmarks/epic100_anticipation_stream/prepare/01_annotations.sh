#!/usr/bin/env bash
# epic100_anticipation_stream — step 1: the annotation csvs at the pinned commit of epic-kitchens-100-annotations.
set -euo pipefail
ROOT="$(yavih config get epic100_anticipation_stream.media_root)"
C=ea8b40457a400c3fffa1c7f406ef3dc169cc2522
mkdir -p "$ROOT/annotations"
cd "$ROOT/annotations"
for f in EPIC_100_train.csv EPIC_100_validation.csv EPIC_100_video_info.csv EPIC_100_verb_classes.csv EPIC_100_noun_classes.csv \
         EPIC_100_tail_verbs.csv EPIC_100_tail_nouns.csv EPIC_100_unseen_participant_ids_validation.csv; do
  [[ -s "$f" ]] || curl -sfL -o "$f" "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/$C/$f"
done
sha256sum -c - <<'SUMS'
a3a7ef2e397bd8af12bd0496b09d61b1f2c1f3f6e1726fa86af50a27f4855e87  EPIC_100_train.csv
35f7932ba0a1127a96cac215a98d35398946f343e3cea9ad6688ed17eee9d75d  EPIC_100_validation.csv
75fd040f6662cb407b4ca2eed4811ca94df280a1aba00b6bf8a5cd286deff382  EPIC_100_video_info.csv
aab8bf210d0234b6facb9b191ae402c03a5f8965d123586b4762d3468a85033e  EPIC_100_verb_classes.csv
40f0a81c5bd06b33050df9f0bd17cd3c2fccd759f9581e4e5e04fec7562be893  EPIC_100_noun_classes.csv
6f124f23759b2bfaa742e071619d78a08b5d2f80147db9919c612c72c9774c94  EPIC_100_tail_verbs.csv
62e3449a7a95fdcb43bc2eaa52a191844f4a566ff1178d2c867b0c0822387266  EPIC_100_tail_nouns.csv
1274797112b20dd821cd10f62537545e669c4b7d742e36590e9f0f7ffb0c6a54  EPIC_100_unseen_participant_ids_validation.csv
SUMS
