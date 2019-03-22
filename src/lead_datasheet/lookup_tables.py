KEY_TO_NAME = {
    7: 'borrow (+7)',
    6: 'borrow (+6)',
    5: 'borrow (+5)',
    4: 'borrow (+4)',
    3: 'borrow (+3)',
    2: 'borrow (+2)',
    1: 'Lydian',
    0: 'Major/Ionian',
    -1: 'Mixolydian',
    -2: 'Dorian',
    -3: 'Minor/Aeolian',
    -4: 'Phrygian',
    -5: 'Locrian',
    -6: 'borrow (-6)',
    -7: 'borrow (-7)'
}


MODE_TO_NAME = {
    1: 'Major/Ionian',
    2: 'Dorian',
    3: 'Phrygian',
    4: 'Lydian',
    5: 'Mixolydian',
    6: 'Minor/Aeolian',
    7: 'Locrian'
}

# [Reference]
# http://forum.hooktheory.com/t/using-chords-in-any-transposition/110/4
#  6 - S(6) Supermode     (F#) - ######
#  5 - S(5) Supermode      (B) - #####
#  4 - S(4) Supermode      (E) - ####
#  3 - S(3) Supermode      (A) - ###
#  2 - S(2) Supermode      (D) - ##
#  1 - Lydian              (G) - #
#  G - Ionian/Major        (C)
# B1 - Mixolydian          (F) - b
# B2 - Dorian             (Bb) - bb
# B3 - Aeolian/Minor      (Eb) - bbb
# B4 - Phrygian           (Ab) - bbbb
# B5 - Locrian            (Db) - bbbbb
# B6 - ???? Supermode     (Gb) - bbbbbb

KEY_TO_SCALE = {
    7:  [1, 3, 5, 6, 8, 10, 12],  # F#
    6:  [1, 3, 5, 6, 8, 10, 11],  # F#
    5:  [1, 3, 4, 6, 8, 10, 11],  # B
    4:  [1, 3, 4, 6, 8, 9, 11],   # E
    3:  [1, 2, 4, 6, 8, 9, 11],   # A
    2:  [1, 2, 4, 6, 7, 9, 11],   # D
    1:  [0, 2, 4, 6, 7, 9, 11],   # G  (Lydian)
    0:  [0, 2, 4, 5, 7, 9, 11],   # C  (Ionian/Major)
    -1: [0, 2, 4, 5, 7, 9, 10],   # F  (Mixolydian)
    -2: [0, 2, 3, 5, 7, 9, 10],   # Bb (Dorian)
    -3: [0, 2, 3, 5, 7, 8, 10],   # Eb (Aeolian/Minor)
    -4: [0, 1, 3, 5, 7, 8, 10],   # Ab (Phrygian)
    -5: [0, 1, 3, 5, 6, 8, 10],   # Db (Locrian)
    -6: [-1, 1, 3, 5, 6, 8, 10],  # Gb
    -7: [-1, 0, 3, 4, 6, 8, 10],  # Gb

    # from michael-jackson/you-are-not-alone/bridge
    'b': [0, 2, 3, 5, 7, 8, 10],   # Eb (Aeolian/Minor)

}

VAL_TO_NAME = {
    0:  ['C', 'C'],
    1:  ['Db', 'C#'],
    2:  ['D', 'D'],
    3:  ['Eb', 'D#'],
    4:  ['E', 'E'],
    5:  ['F', 'F'],
    6:  ['Gb', 'F#'],
    7:  ['G', 'G'],
    8:  ['Ab', 'G#'],
    9:  ['A', 'A'],
    10: ['Bb', 'A#'],
    11: ['B', 'B']
}

NOTE_TO_OFFSET = {
    'Ab': 8,
    'A': 9,
    'A#': 10,
    'Bb': 10,
    'B': 11,
    'Cb': 11,
    'C': 0,
    'C#': 1,
    'Db': 1,
    'D': 2,
    'D#': 3,
    'Eb': 3,
    'E': 4,
    'E#': 5,  # knife-party/power-glove/intro
    'F': 5,
    'F#': 6,
    'Gb': 6,
    'G': 7,
    'G#': 8,
}

MODE_TO_KEY = {
    1: 0,
    2: -2,
    3: -4,
    4: 1,
    5: -1,
    6: -3,
    7: -5
}
# KEY_DICT map the key string to its note number
KEY_DICT = {
    'A': 57,
    'B': 59,
    'C': 60,
    'D': 62,
    'E': 64,
    'F': 65,
    'G': 67
}

# ACCIDENTAL_DICT map the accidental string to its corresponding value
ACCIDENTAL_DICT = {
    '#': 1,
    'b': -1,
    's': 1,
    'f': -1
}


def get_key_root(key):
    """Given the key string (e.g. 'C', 'F#', 'Db'), return the note number
    of the root note"""
    if len(key) > 1:
        return KEY_DICT[key[0]] + ACCIDENTAL_DICT[key[1]]
    else:
        return KEY_DICT[key[0]]
