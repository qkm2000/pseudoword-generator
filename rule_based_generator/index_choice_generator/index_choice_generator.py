import random


random.seed(42)


CLIP_CONSONANTS = ['x', 'h', 'k', 't', 'g', 's', 'p', 'n', 'd', 'l', 'b', 'm']
CLIP_VOWELS = ['i', 'e', 'a', 'o', 'u']

SD_CONSONANTS = ['x', 'h', 't', 'k', 'p', 's', 'g', 'd', 'l', 'n', 'm', 'b']
SD_VOWELS = ['e', 'i', 'a', 'o', 'u']

SYLLABLE_STRUCTURES = [
    'cvc',
    'vcv',
    'cvv',
    'cvcv',
    'cvvc',
    'vccv',
    'vcvc',
    'cvvcv',
    'vcvvc',
    'cvcvc',
    'vcvcv'
]


def generate_structure():
    return random.choice(SYLLABLE_STRUCTURES)


def generate_pseudoword(input_val, consonants, vowels):
    input_val = max(0.0, min(1.0, input_val))
    structure = generate_structure()

    c_idx = round(input_val * (len(consonants) - 1))
    v_idx = round(input_val * (len(vowels) - 1))

    c_idx = max(0, min(len(consonants) - 1, c_idx))
    v_idx = max(0, min(len(vowels) - 1, v_idx))

    c_idx = [c_idx-2, c_idx+2]
    v_idx = [v_idx-1, v_idx+1]

    if c_idx[0] < 0:
        c_idx[0] = 0
    if c_idx[1] >= len(consonants):
        c_idx[1] = len(consonants)
    if v_idx[0] < 0:
        v_idx[0] = 0
    if v_idx[1] >= len(vowels):
        v_idx[1] = len(vowels)

    # print(f"c_idx: {c_idx}")
    # print(f"v_idx: {v_idx}")

    chosen_c = consonants[c_idx[0]:c_idx[1]]
    chosen_v = vowels[v_idx[0]:v_idx[1]]

    # print(f"chosen_c: {chosen_c}")
    # print(f"chosen_v: {chosen_v}")

    pseudoword = []
    for s in structure:
        pseudoword.append(
            random.choice(chosen_c) if s == 'c' else random.choice(chosen_v)
        )

    return {
        'consonants': chosen_c,
        'vowels': chosen_v,
        'pseudoword': ''.join(pseudoword)
    }


def main(clip=False, sd=False):
    if sd:
        consonants = SD_CONSONANTS
        vowels = SD_VOWELS
        model = 'Stable Diffusion'
    if clip:
        consonants = CLIP_CONSONANTS
        vowels = CLIP_VOWELS
        model = 'CLIP'
    print(f"using {model}'s consonants and vowels")
    num_steps = 11
    roundness_values = [i/(num_steps-1) for i in range(num_steps)]
    for i in range(num_steps):
        input_value = roundness_values[i]
        result = generate_pseudoword(input_value, consonants, vowels)
        print(f"Roundness: {input_value:.4f}", end=" | ")
        print(f"Pseudoword: {result['pseudoword']:<8}", end=" | ")
        print(f"Consonants: {str(result['consonants']):<20}", end=" | ")
        print(f"Vowels: {result['vowels']}")


if __name__ == '__main__':
    main(
        clip=True
    )
    print()
    main(
        sd=True
    )
