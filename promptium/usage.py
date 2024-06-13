
from dataclasses import dataclass


OpenAI = ...
ChatGPT = ...

account = OpenAI('~/.keys', total_call_limit=1000)
chatGPT = ChatGPT(account, temperature=1.5)

@chatGPT.prompt
@dataclass
class GenAnimals:
    n: int
    animals: list = None
    llm: ChatGPT = None
    def __call__(self):
        """
        Write a comma-separated list of {n} animals.
        """
        if self.n > 0:
            gen = self.llm(self.n)
            return gen.split(',')
        self.animals = []

animals = GenAnimals(10)
print(f'Generated {animals.n} animals:')
for animal in animals.animals:
    print(animal)

@chatGPT.prompt(max_tokens=200)
def gen_animals(n, llm=None):
    """
    Write a comma-separated list of {n} animals.
    """
    if n > 0:
        gen = llm(n)
        return gen.split(',')
    return []

@chatGPT
def gen_related_animals(animal, n, gen=None):
    """
    Write a list of {n} animals related to a {animal}.

    |1.
    """
    return gen.split('\n')

@chatGPT
def gen_description(animal):
    """
    Tell me about {animal}.


    """


def animal_pipeline():
    for animal in gen_animals(10):
        for related in gen_related_animals(animal, 3):
            print(related, 'is related to', animal)
            description = gen_description(related)
            print(description)


def fast_animal_pipeline():
    animals = dict.fromkeys(gen_animals(10))
    related_animals = gen_related_animals.batch(
        gen_related_animals(animal, 3) for animal in animals
    )
    animals = dict(zip(animals, related_animals))
    descriptions = gen_description.batch(
        gen_description(related)
        for animal, related in animals.items()
        for related_animal in related
    )


send_to_gpt = ...


def super_fast_animal_pipeline():
    results = []
    for result in send_to_gpt(gen_animals(10)):
        if result.prompt is gen_animals:
            animals = result.output
            for animal in animals:
                send_to_gpt(gen_related_animals(animal, 3))
        if result.prompt is gen_related_animals:
            related_animals = result.output
            for related in related_animals:
                send_to_gpt(gen_description(related))
        if result.prompt is gen_description:
            results.append(result)
    for result in results:
        animal = result.animal
        related = result.related
        description = result.output



