import json
import random


random.seed(42)


def load_sound_mappings(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def select_sounds(
    bouba_kiki_value,
    sound_dict,
    num_sounds=5
):
    sorted_sounds = sorted(
        sound_dict.items(),
        key=lambda x: abs(x[1] - bouba_kiki_value)
    )
    return [sound[0] for sound in sorted_sounds[:num_sounds]]


def generate_pseudoword(
    consonants,
    vowels,
):
    syllable_structures = [
        "cvc",
        "cvv",
        "vvc",
        "cvcv",
        "cvcvv",
        "cvvcv",
        "vcv",
        "vvcv"
    ]
    pseudoword = ""
    structure = random.choice(syllable_structures)
    for char in structure:
        if char == "c":
            pseudoword += random.choice(consonants)
        else:
            pseudoword += random.choice(vowels)
    return pseudoword


def evaluate_pseudoword(
    pseudoword,
    bouba_kiki_value,
    filename='sound_mappings.json'
):
    sound_dict = load_sound_mappings(filename)
    total_bouba_kiki_value = 0
    for char in pseudoword:
        if char in sound_dict["vowels"]:
            total_bouba_kiki_value += sound_dict["vowels"][char]
        else:
            total_bouba_kiki_value += sound_dict["consonants"][char]
    avg_bouba_kiki_value = total_bouba_kiki_value / len(pseudoword)

    error = abs(avg_bouba_kiki_value - bouba_kiki_value)
    return avg_bouba_kiki_value, error


# Main function
def pseudoword_generator(
    bouba_kiki_value,
    filename='utils/sound_mappings.json',
    sound_dict=None
):
    if sound_dict is None:
        sound_dict = load_sound_mappings(filename)
    selected_consonants = select_sounds(
        bouba_kiki_value,
        sound_dict["consonants"]
    )
    selected_vowels = select_sounds(
        bouba_kiki_value,
        sound_dict["vowels"],
        num_sounds=3
    )

    return generate_pseudoword(
        selected_consonants,
        selected_vowels,
    )


if __name__ == "__main__":
    filename = r'datasets\sound_mappings.json'
    num_tests = 11
    bouba_kiki_value = [i/(num_tests-1) for i in range(num_tests)]
    scores = []
    accumulated_error = 0

    for i in range(num_tests):
        pseudoword = pseudoword_generator(
            bouba_kiki_value[i],
            filename
        )
        score, error = evaluate_pseudoword(
            pseudoword,
            bouba_kiki_value[i],
            filename
        )
        scores.append(score)
        print(f"Roundness: {bouba_kiki_value[i]:.4f}", end=" | ")
        print(f"Pseudoword: {pseudoword:<8}", end=" | ")
        print(f"Score: {score:.4f}", end=" | ")
        print(f"Error: {error:.4f}")
        accumulated_error += error

    print(f"\nAverage error: {accumulated_error / num_tests:.4f}")
