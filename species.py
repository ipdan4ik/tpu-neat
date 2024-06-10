from genome import Genome


class Species:
    def __init__(self, representative):
        self.representative: Genome = representative
        self.genomes: list[Genome] = []
        self.adjusted_fitness_sum: float = 0
        self.avg_adjusted_fitness: float = 0
        self.max_fitness: float = 0
        self.prev_max_fitness: float = 0
        self.age: int = 0
        self.no_improvement_age: int = 0
        self.has_best: bool = False
        self.offspring_number: int = 0

    def add_genome(self, genome: Genome):
        self.genomes.append(genome)
        self.recalculate_adjusted_fitness()
        self.adjusted_fitness_sum = sum([g.adjusted_fitness for g in self.genomes])
        self.avg_adjusted_fitness = self.adjusted_fitness_sum / len(self.genomes)

    def recalculate_adjusted_fitness(self):
        for genome in self.genomes:
            genome.adjusted_fitness = genome.fitness / len(self.genomes)

    def full_recalculate(self):
        self.recalculate_adjusted_fitness()
        self.adjusted_fitness_sum = sum([g.adjusted_fitness for g in self.genomes])
        self.avg_adjusted_fitness = self.adjusted_fitness_sum / len(self.genomes)

    def recalculate_max_fitness(self):
        self.prev_max_fitness = self.max_fitness
        self.max_fitness = max(self.genomes, key=lambda genome: genome.fitness).fitness

    def evolve_step(self):
        self.genomes = []
        self.adjusted_fitness_sum = 0
        self.avg_adjusted_fitness = 0
        self.age += 1
