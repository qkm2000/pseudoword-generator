import random

# note: all of the consonants and vowels are based on the paper
# https://arxiv.org/html/2310.16781v2

# default consonants
DEFAULT_CONSONANTS = {
    0: ['p', 't', 'k', 's', 'h', 'x'],
    1: ['b', 'd', 'g', 'm', 'n', 'l'],
}

# stable diffusion's consonants
SD_CONSONANTS = {
    0:   ['x', 'h', 't', 'k'],
    0.5: ['p', 's', 'g', 'd'],
    1:   ['l', 'n', 'm', 'b'],
}

# CLIP's consonants
CLIP_CONSONANTS = {
    0:   ['x', 'h', 'k', 't'],
    0.5: ['g', 's', 'p', 'n'],
    1:   ['d', 'l', 'b', 'm'],
}

# default vowels
DEFAULT_VOWELS = {
    0:   ['e', 'i'],
    0.5: ['a'],
    1:   ['o', 'u'],
}

# these vowels are based on stable diffusion
# SD_VOWELS_3 takes the vowels from the paper
# and adds the previous and next vowel
SD_VOWELS_3 = {
    0:    ['i', 'e'],
    0.25: ['i', 'e', 'a'],
    0.5:  ['e', 'a', 'o'],
    0.75: ['a', 'o', 'u'],
    1:    ['o', 'u']
}

# these vowels are based on CLIP
# CLIP_VOWELS_3 takes the vowels from the paper
# and adds the previous and next vowel
CLIP_VOWELS_3 = {
    0: ['e', 'i'],
    0.25: ['i', 'e', 'a'],
    0.5: ['e', 'a', 'o'],
    0.75: ['a', 'o', 'u'],
    1: ['o', 'u']
}

# Define syllable structures
SYLLABLE_STRUCTURES = {
    0: 'CVC',
    0.25: 'CVCV',
    0.5: 'CVCCV',
    0.75: 'CVVCV',
    1: 'CVVCVC'
}


def get_closest_key(value, dictionary):
    '''
    Find the closest key in a dictionary to the given value.
    '''
    return min(dictionary.keys(), key=lambda k: abs(k - value))


def generate_pseudoword(roundness, consonants=None, vowels=None):
    '''
    Generate a single pseudoword for the given roundness,
    using the selected consonant and vowel groups.
    Returns the pseudoword, consonant group, vowel group, and structure.
    '''
    if roundness < 0 or roundness > 1:
        raise ValueError('Roundness must be between 0 and 1.')

    if consonants is None or vowels is None:
        raise ValueError('Consonants and vowels must be provided.')

    # Get the closest key for consonants, vowels, and syllable structure
    consonant_key = get_closest_key(roundness, consonants)
    vowel_key = get_closest_key(roundness, vowels)
    structure_key = get_closest_key(roundness, SYLLABLE_STRUCTURES)

    # Select consonants, vowels, and syllable structure
    consonants_list = consonants[consonant_key]
    vowels_list = vowels[vowel_key]
    structure = SYLLABLE_STRUCTURES[structure_key]

    # Build the pseudoword by selecting random consonants
    # and vowels for the structure
    pseudoword = ""
    for char in structure:
        if char == 'C':
            pseudoword += random.choice(consonants_list)
        elif char == 'V':
            pseudoword += random.choice(vowels_list)

    return pseudoword, consonants_list, vowels_list, structure


# Example usage
if __name__ == '__main__':
    # Define combinations of consonant/vowel groups to test
    group_combinations = [
        (DEFAULT_CONSONANTS, DEFAULT_VOWELS),
        (SD_CONSONANTS, SD_VOWELS_3),
        (CLIP_CONSONANTS, CLIP_VOWELS_3)
    ]

    # Test with roundness values
    num_steps = 11
    roundness_values = [i/(num_steps-1) for i in range(num_steps)]
    # roundness_values = [0, 0.25, 0.5, 0.75, 1]
    for value in roundness_values:
        for cons, vows in group_combinations:
            # Generate pseudoword for this group combination
            pseudoword, cons_group, vow_group, structure = generate_pseudoword(
                value,
                cons,
                vows
            )
            # print(f"  Structure: {structure}")
            print(f"Roundness: {value:.4f}", end=" | ")
            print(f"Pseudoword: {pseudoword:<8}", end=" | ")
            print(f'Consonant Group: {str(cons_group):<30}', end=" | ")
            print(f'Vowel Group: {vow_group}')
        print()
