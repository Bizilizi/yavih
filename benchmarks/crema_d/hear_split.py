# ruff: noqa: UP004, UP031, UP032, UP035, B007, B905  -- the copied upstream code keeps its own style (typing aliases, format calls, zip)
"""HEAR's five CREMA-D folds, re-derived from the public CREMA-D files without TensorFlow.

HEAR 2021 (``hearbenchmark/hear-preprocess`` task ``tfds_crema_d``, ``split_mode: new_split_kfold``, ``nfolds: 5``)
builds CREMA-D from the TensorFlow Datasets builder ``crema_d`` 1.0.0, names every clip after its ``tfds_id``
(``tfds_id_<slugify(tfds_id)>``) and splits the sorted names into five folds with a seeded shuffle. A clip's fold
therefore depends on where TFDS wrote it: its TFDS split (a speaker-grouped 70/10/20 draw), its position after the
TFDS writer's hash shuffle, and the number of shards the writer chose from the serialized size of the split. This
module re-computes each of these steps in plain Python and numpy, from the CREMA-D ``summaryTable.csv`` and the
number of samples and the varint size of the 16-bit PCM samples of each wav (``make_manifest.py`` reads them from
the staged wavs; a ``--no-media`` re-freeze reads the per-split shard counts pinned there). Module scope imports the
standard library and numpy only.

Copied code, every block marked below with its source:

* TFDS (``tensorflow/datasets``, tag ``v4.4.0`` = commit ``76f8591def26afaca16340b06d057553582f6163``, Apache-2.0):
  ``tensorflow_datasets/audio/crema_d.py`` (``LABELS``, the bad files of ``_split_generators``,
  ``_compute_split_boundaries``, ``_get_inter_splits_by_group`` and the example key of ``_generate_examples``),
  ``tensorflow_datasets/core/hashing.py`` (``Hasher``), ``tensorflow_datasets/core/shuffle.py`` (the sort by
  hashed key of ``Shuffler``), ``tensorflow_datasets/core/tfrecords_writer.py`` (``_get_number_shards``,
  ``_get_shard_boundaries``, the shard file name) and ``tensorflow_datasets/core/tfrecords_reader.py``
  (``_make_id``: ``<shard file name>__<index in shard>``). Changes: TensorFlow, ``tf.io.gfile`` and the download
  manager removed; the functions take the names and sizes as arguments; ``Hasher`` keeps only ``hash_key`` for
  ``str`` keys.
* hear-preprocess (``hearbenchmark/hear-preprocess`` @ ``610d864b2c0f7064eaeced98ee8992d3914d58f1``, v2021.1.1,
  Apache-2.0): ``hearpreprocess/pipeline.py`` ``ExtractMetadata.split_k_folds`` and the file stem of
  ``hearpreprocess/tfds_pipeline.py`` ``ExtractTFDS.save_audio_labels`` (``f"tfds_id_{slugify(tfds_id)}"``).
  Changes: pandas replaced by lists and dicts (same sort, same ``random.Random`` shuffle, same
  ``np.array_split``, same label check per fold); the ``small`` mode branch removed.
* python-slugify (``un33k/python-slugify`` v6.1.2 = ``3f1a0fe7c5775a72141163ccdd593272e512898c``, MIT):
  ``slugify/slugify.py`` ``slugify`` reduced to what it does to an ASCII ``tfds_id``
  (lower case, every run of characters outside ``[-a-z0-9]`` to one ``-``, ``-`` runs collapsed, ``-`` stripped).

Written here (no implementation to copy): the size of a serialized ``tf.train.Example`` of one CREMA-D clip, from
the protobuf wire format (:func:`example_size`), which TFDS takes from ``SerializeToString()``.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import random
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------- TFDS audio/crema_d.py (v4.4.0)
# Copied from tensorflow_datasets/audio/crema_d.py, Apache-2.0, unchanged unless noted.

LABELS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']

# From CremaD._split_generators: "These are file names which do do not exist in the github"
BAD_FILES = frozenset([
    'FileName', '1040_ITH_SAD_XX', '1006_TIE_NEU_XX', '1013_WSI_DIS_XX',
    '1017_IWW_FEA_XX'
])

SPLIT_PROBS = [('train', 0.7), ('validation', 0.1), ('test', 0.2)]


def summary_table_names(lines: Iterable[str]) -> list[str]:
  """The wav names CremaD._split_generators keeps, in summaryTable.csv order (change: a function over the lines)."""
  names = []
  for line in lines:
    wav_name = line.strip().split(',')[1].replace('"', '')
    if (not wav_name) or (wav_name in BAD_FILES):
      continue
    names.append(wav_name)
  return names


def _compute_split_boundaries(split_probs, n_items):
  """Computes boundary indices for each of the splits in split_probs.

  Args:
    split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
      ('test', 0.2)]
    n_items: Number of items we want to split.

  Returns:
    The item indices of boundaries between different splits. For the above
    example and n_items=100, these will be
    [('train', 0, 60), ('dev', 60, 80), ('test', 80, 100)].
  """
  if len(split_probs) > n_items:
    raise ValueError('Not enough items for the splits. There are {splits} '
                     'splits while there are only {items} items'.format(
                         splits=len(split_probs), items=n_items))
  total_probs = sum(p for name, p in split_probs)
  if abs(1 - total_probs) > 1E-8:
    raise ValueError('Probs should sum up to 1. probs={}'.format(split_probs))
  split_boundaries = []
  sum_p = 0.0
  for name, p in split_probs:
    prev = sum_p
    sum_p += p
    split_boundaries.append((name, int(prev * n_items), int(sum_p * n_items)))

  # Guard against rounding errors.
  split_boundaries[-1] = (split_boundaries[-1][0], split_boundaries[-1][1],
                          n_items)

  return split_boundaries


def _get_inter_splits_by_group(items_and_groups, split_probs, split_number):
  """Split items to train/dev/test, so all items in group go into same split.

  Each group contains all the samples from the same speaker ID. The samples are
  splitted between train, validation and testing so that samples from each
  speaker belongs to exactly one split.

  Args:
    items_and_groups: Sequence of (item_id, group_id) pairs.
    split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
      ('test', 0.2)]
    split_number: Generated splits should change with split_number.

  Returns:
    Dictionary that looks like {split name -> set(ids)}.
  """
  groups = sorted(set(group_id for item_id, group_id in items_and_groups))
  rng = np.random.RandomState(split_number)
  rng.shuffle(groups)

  split_boundaries = _compute_split_boundaries(split_probs, len(groups))
  group_id_to_split = {}
  for split_name, i_start, i_end in split_boundaries:
    for i in range(i_start, i_end):
      group_id_to_split[groups[i]] = split_name

  split_to_ids = collections.defaultdict(set)
  for item_id, group_id in items_and_groups:
    split = group_id_to_split[group_id]
    split_to_ids[split].add(item_id)

  return split_to_ids


def tfds_splits(wav_names: Sequence[str]) -> dict[str, set[str]]:
  """CremaD._split_generators: speaker = the name's first field, split_number 0 (change: names, not paths)."""
  speaker_ids = [wav_name.split('_')[0] for wav_name in wav_names]
  return _get_inter_splits_by_group(list(zip(wav_names, speaker_ids)), SPLIT_PROBS, 0)


def tfds_label(wav_name: str) -> str:
  """CremaD._generate_examples: ``label = file_name.split('_')[2]``; the example key is ``file_name``."""
  return wav_name.split('_')[2]


# --------------------------------------------------------------------------- TFDS core (v4.4.0)
# hashing.py: Hasher (md5 salted with the split name; change: str keys only).


class Hasher(object):
  """Hasher: to initialize a md5 with salt."""

  def __init__(self, salt):
    self._md5 = hashlib.md5(salt.replace('\\', '/').encode('utf-8'))

  def hash_key(self, key):
    """Returns 128 bits hash of given key."""
    md5 = self._md5.copy()
    md5.update(key.replace('\\', '/').encode('utf-8'))
    return int(md5.hexdigest(), 16)


# tfrecords_writer.py: shard sizing (MIN/MAX_SHARD_SIZE, TFRECORD_REC_OVERHEAD, _get_number_shards,
# _get_shard_boundaries), unchanged.
MIN_SHARD_SIZE = 64 << 20  # 64 MiB
MAX_SHARD_SIZE = 1024 << 20  # 2 GiB

TFRECORD_REC_OVERHEAD = 16


def _get_shard_boundaries(
    num_examples: int,
    number_of_shards: int,
) -> list[int]:
  if num_examples == 0:
    raise AssertionError("No examples were yielded.")
  if num_examples < number_of_shards:
    raise AssertionError("num_examples ({}) < number_of_shards ({})".format(
        num_examples, number_of_shards))
  return [
      int(round(num_examples * (float(i) / number_of_shards)))
      for i in range(1, number_of_shards + 1)
  ]


def _get_number_shards(
    total_size: int,
    num_examples: int,
) -> int:
  """Returns number of shards for num_examples of total_size in bytes."""
  total_size += num_examples * TFRECORD_REC_OVERHEAD
  max_shards_number = total_size // MIN_SHARD_SIZE
  min_shards_number = total_size // MAX_SHARD_SIZE
  if min_shards_number <= 1024 <= max_shards_number and num_examples >= 1024:
    return 1024
  elif min_shards_number > 1024:
    i = 2
    while True:
      n = 1024 * i
      if n >= min_shards_number and num_examples >= n:
        return n
      i += 1
  else:
    for n in [512, 256, 128, 64, 32, 16, 8, 4, 2]:
      if min_shards_number <= n <= max_shards_number and num_examples >= n:
        return n
  return 1


def tfds_ids(split: str, keys: Sequence[str], num_shards: int, *, dataset: str = 'crema_d') -> dict[str, str]:
  """``{example key: tfds_id}`` of one split: the Shuffler's order (``sorted`` by ``Hasher(split).hash_key``,
  shuffle.py ``_iter_mem``; the buckets of a larger split keep the same global order), cut by
  ``_get_shard_boundaries`` into ``<dataset>-<split>.tfrecord-%05d-of-%05d`` files, and the reader's
  ``_make_id`` (``<file name>__<index>``)."""
  hasher = Hasher(split)
  order = sorted(keys, key=hasher.hash_key)
  hkeys = [hasher.hash_key(k) for k in order]
  if len(set(hkeys)) != len(hkeys):
    raise AssertionError("Two examples share the same hashed key!")
  out: dict[str, str] = {}
  from_ = 0
  for shard_index, to in enumerate(_get_shard_boundaries(len(order), num_shards)):
    filename = "%s-%s.tfrecord-%05d-of-%05d" % (dataset, split, shard_index, num_shards)
    for index, key in enumerate(order[from_:to]):
      out[key] = f"{filename}__{index}"
    from_ = to
  return out


# --------------------------------------------------------------------------- the serialized size (written here)


def _varint_size(v: int) -> int:
  """Bytes of a protobuf varint of a non-negative int (int64 fields: a negative value is 10 bytes)."""
  n = 1
  while v >= 0x80:
    v >>= 7
    n += 1
  return n


def packed_int64_payload(samples: np.ndarray) -> int:
  """Bytes of the packed ``int64`` values of ``samples`` (the 16-bit PCM samples TFDS stores as int64)."""
  s = np.asarray(samples).astype(np.int64, copy=False).ravel()
  neg = int(np.count_nonzero(s < 0))
  pos = s[s >= 0]
  return (neg * 10 + int(pos.size) + int(np.count_nonzero(pos >= 1 << 7)) + int(np.count_nonzero(pos >= 1 << 14))
          + int(np.count_nonzero(pos >= 1 << 21)))


def example_size(n_samples: int, payload: int, label_index: int, speaker_id: str) -> int:
  """``len(tf.train.Example(features={audio: int64_list, label: int64_list, speaker_id: bytes_list}).SerializeToString())``
  for one CREMA-D clip of ``n_samples`` samples whose packed values take ``payload`` bytes."""
  def field(size: int) -> int:  # a length-delimited field: tag + length + bytes
    return 1 + _varint_size(size) + size

  def feature_int64(n: int, p: int) -> int:  # Feature{int64_list: Int64List{packed value}}
    int64_list = field(p) if n else 0
    return field(int64_list)

  def feature_bytes(values: Sequence[bytes]) -> int:  # Feature{bytes_list: BytesList{value}}
    return field(sum(field(len(v)) for v in values))

  def entry(key: str, feature_size: int) -> int:  # Features.feature map entry {1: key, 2: Feature}
    return field(field(len(key.encode())) + field(feature_size))

  features = (entry('audio', feature_int64(n_samples, payload)) + entry('label', feature_int64(1, _varint_size(label_index)))
              + entry('speaker_id', feature_bytes([speaker_id.encode()])))
  return field(features)


# --------------------------------------------------------------------------- python-slugify (ASCII inputs)


def slugify(text: str) -> str:
  text = text.lower()
  text = re.sub(r'[^-a-z0-9]+', '-', text)
  text = re.sub(r'-{2,}', '-', text)
  return text.strip('-')


def hear_filestem(tfds_id: str) -> str:
  """hear-preprocess tfds_pipeline.ExtractTFDS.save_audio_labels: ``f"tfds_id_{slugify(tfds_id)}.wav"`` (the stem)."""
  return f"tfds_id_{slugify(tfds_id)}"


# --------------------------------------------------------------------------- hear-preprocess pipeline.py


def split_k_folds(split_key_label: Sequence[tuple[str, str]], k_folds: int = 5) -> dict[str, str]:
  """ExtractMetadata.split_k_folds: ``{split_key: "fold%02d"}`` (change: (split_key, label) pairs instead of the
  metadata frame; the default split_key of a scene task is the unique file stem, one row per file)."""
  SEED_RETRIES = 1000
  # Deterministically sort all unique split_keys.
  split_keys = sorted({k for k, _ in split_key_label})
  labels = {lab for _, lab in split_key_label}

  # A set of seeds will be tried to ensure each fold has at least
  # one example for each label
  # Since seeds are tried sequentially, this is still deterministic
  # The first seed remains "split_k_folds" to maintain consistency with
  # previous versions
  seeds = ["split_k_folds"] + [
      f"split_k_folds_retry_{i}" for i in range(SEED_RETRIES - 1)
  ]

  fold_of: dict[str, str] = {}
  for i, seed in enumerate(seeds):
    # Deterministically shuffle all unique split_keys.
    rng = random.Random(seed)
    shuffled_split_keys = copy.deepcopy(split_keys)
    rng.shuffle(shuffled_split_keys)

    # Equally split the split_keys into k folds and label accordingly
    folds_keys = np.array_split(shuffled_split_keys, k_folds)
    fold_of = {}
    for j, fold in enumerate(folds_keys):
      for key in fold.tolist():
        fold_of[key] = f"fold{j:02d}"

    per_fold: dict[str, set] = collections.defaultdict(set)
    for key, lab in split_key_label:
      per_fold[fold_of[key]].add(lab)
    if all(per_fold[f"fold{j:02d}"] == labels for j in range(k_folds)):
      break

  if i == len(seeds) - 1:
    raise AssertionError(
        "All the seeds were tried, but the data couldnot be split into "
        "folds with at least one example for each label"
    )
  return fold_of


# --------------------------------------------------------------------------- the whole derivation


def derive(wav_names: Sequence[str], num_shards: Mapping[str, int]) -> list[dict[str, str]]:
  """Every kept clip in summaryTable.csv order: ``{wav_name, tfds_split, tfds_id, hear_stem, fold, label}``."""
  splits = tfds_splits(wav_names)
  if set(splits) != {s for s, _ in SPLIT_PROBS}:
    raise AssertionError(f"TFDS splits {sorted(splits)}")
  ids: dict[str, tuple[str, str]] = {}
  for split, keys in splits.items():
    for key, tid in tfds_ids(split, sorted(keys), int(num_shards[split])).items():
      ids[key] = (split, tid)
  stems = {name: hear_filestem(ids[name][1]) for name in wav_names}
  if len(set(stems.values())) != len(stems):
    raise AssertionError("HEAR file stems are not unique")
  folds = split_k_folds([(stems[n], tfds_label(n)) for n in wav_names])
  return [{"wav_name": n, "tfds_split": ids[n][0], "tfds_id": ids[n][1], "hear_stem": stems[n], "fold": folds[stems[n]],
           "label": tfds_label(n)} for n in wav_names]
