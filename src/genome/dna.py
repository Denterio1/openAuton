"""
Cognitive DNA & Evolution Engine

This module implements a genetic-evolutionary framework for agent self-improvement.
Genes represent reusable patterns (architectures, hyperparameters, training strategies)
extracted from experience episodes. CognitiveDNA is a genome that encodes an agent's
"knowledge" and "strategies". The EvolutionEngine evolves a population of DNAs
through selection, crossover, and mutation, driving continuous improvement.

Inspired by:
- Biological evolution (selection, mutation, crossover)
- Memetics (cultural evolution)
- Meta-learning (learning to learn)
- Genetic algorithms for hyperparameter optimization
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from enum import Enum
import hashlib
import uuid
import json
from pathlib import Path

# ============= GENE TYPES =============

class GeneType(Enum):
    """Types of genes in the cognitive DNA"""
    ARCHITECTURE = "architecture"           # Model architecture choice
    HYPERPARAMETER = "hyperparameter"       # Learning rate, batch size, etc.
    DATA_STRATEGY = "data_strategy"         # Synthetic data, augmentation, etc.
    OBJECTIVE = "objective"                 # Loss function, training objective
    TRAINING_SCHEDULE = "training_schedule" # LR schedule, warmup, etc.
    EVALUATION_STRATEGY = "evaluation"      # Metrics, reasoning checks
    RAG_STRATEGY = "rag"                    # Retrieval-augmented generation config
    FINE_TUNING_STRATEGY = "finetuning"     # LoRA, full, etc.
    ENSEMBLE_STRATEGY = "ensemble"          # Model ensemble method
    META_STRATEGY = "meta"                  # How to learn/adapt

class GeneDominance(Enum):
    """How a gene interacts with others"""
    DOMINANT = "dominant"       # Overrides conflicting genes
    RECESSIVE = "recessive"     # Only expressed if no dominant present
    CODOMINANT = "codominant"   # Blends with other genes
    EPISTATIC = "epistatic"     # Modifies expression of other genes

# ============= GENE CLASS =============

@dataclass
class Gene:
    """
    A single gene representing a reusable pattern or strategy.
    
    Each gene has a type, value, confidence, and metadata about its origin
    and performance. It can be dominant, recessive, etc.
    """
    gene_id: str
    name: str
    gene_type: GeneType
    value: Any
    confidence: float                     # 0-1, how reliable this gene is
    dominance: GeneDominance = GeneDominance.DOMINANT
    source_episodes: List[str] = field(default_factory=list)  # episode IDs that contributed
    fitness: float = 0.0                  # Tracked fitness over generations
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.gene_id:
            # Generate deterministic ID based on content
            content = f"{self.name}:{self.gene_type.value}:{str(self.value)}"
            self.gene_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2) -> 'Gene':
        """
        Create a mutated copy of this gene.
        
        Mutation behavior depends on gene type:
        - For numeric values: add Gaussian noise scaled by mutation_strength
        - For categorical: switch to another category with probability mutation_rate
        - For complex objects: apply custom mutation based on type
        """
        import copy
        new_gene = copy.deepcopy(self)
        
        # Numeric values
        if isinstance(self.value, (int, float)):
            # Add noise proportional to value magnitude
            noise = np.random.normal(0, mutation_strength * abs(self.value) if self.value != 0 else mutation_strength)
            new_gene.value = self.value + noise
            # Ensure positive for hyperparameters that require it
            if isinstance(self.value, int):
                new_gene.value = int(max(1, new_gene.value))
            else:
                new_gene.value = max(1e-8, new_gene.value)
        
        # Categorical (using metadata for options if available)
        elif isinstance(self.value, str):
            # If we have possible values in metadata, switch
            options = self.metadata.get("options", None)
            if options and random.random() < mutation_rate:
                # Remove current from options and pick random
                options = [opt for opt in options if opt != self.value]
                if options:
                    new_gene.value = random.choice(options)
        
        # List values (e.g., diversity metrics)
        elif isinstance(self.value, list):
            if random.random() < mutation_rate:
                # Randomly add, remove, or replace an element
                if self.value and random.random() < 0.5:
                    # Remove random element
                    idx = random.randint(0, len(self.value)-1)
                    new_gene.value = self.value[:idx] + self.value[idx+1:]
                else:
                    # Add a new element from possible options
                    options = self.metadata.get("options", [])
                    if options:
                        new_option = random.choice(options)
                        if new_option not in new_gene.value:
                            new_gene.value.append(new_option)
        
        # Update metadata to reflect mutation
        new_gene.metadata["mutated_from"] = self.gene_id
        new_gene.gene_id = str(uuid.uuid4())[:16]
        new_gene.metadata["mutation_time"] = datetime.now().isoformat()
        new_gene.confidence *= (1 - mutation_rate)  # Reduce confidence slightly
        return new_gene
    
    def combine(self, other: 'Gene', method: str = "blend") -> 'Gene':
        """
        Combine this gene with another to create a new gene.
        
        Methods:
        - "blend": average numeric values, random choice for categorical
        - "crossover": take parts from both (for structured values)
        """
        if self.gene_type != other.gene_type:
            raise ValueError("Cannot combine genes of different types")
        
        # Create new gene with combined confidence
        new_confidence = (self.confidence + other.confidence) / 2
        
        if method == "blend":
            if isinstance(self.value, (int, float)):
                new_value = (self.value + other.value) / 2
                if isinstance(self.value, int):
                    new_value = int(new_value)
            elif isinstance(self.value, str):
                # Randomly pick one
                new_value = random.choice([self.value, other.value])
            elif isinstance(self.value, list):
                # Union of both lists
                new_value = list(set(self.value + other.value))
            else:
                # Default: pick self
                new_value = self.value
        else:
            # For now just use blend
            new_value = self.value
        
        new_gene = Gene(
            gene_id="",
            name=f"{self.name}_{other.name}_combo",
            gene_type=self.gene_type,
            value=new_value,
            confidence=new_confidence,
            dominance=GeneDominance.CODOMINANT,
            source_episodes=list(set(self.source_episodes + other.source_episodes)),
            metadata={
                "combined_from": [self.gene_id, other.gene_id],
                "combination_method": method
            }
        )
        return new_gene
    
    def to_dict(self) -> Dict:
        return {
            "gene_id": self.gene_id,
            "name": self.name,
            "gene_type": self.gene_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "dominance": self.dominance.value,
            "source_episodes": self.source_episodes,
            "fitness": self.fitness,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Gene':
        return cls(
            gene_id=data["gene_id"],
            name=data["name"],
            gene_type=GeneType(data["gene_type"]),
            value=data["value"],
            confidence=data["confidence"],
            dominance=GeneDominance(data["dominance"]),
            source_episodes=data.get("source_episodes", []),
            fitness=data.get("fitness", 0.0),
            metadata=data.get("metadata", {})
        )

# ============= COGNITIVE DNA =============

@dataclass
class CognitiveDNA:
    """
    A genome: a collection of genes that define an agent's "knowledge" and strategies.
    
    DNA can evolve through mutation and crossover, and its fitness is measured
    by the performance of the resulting models/strategies.
    """
    dna_id: str = field(
        default_factory=lambda: hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:16])
    genes: List[Gene] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.now)
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.dna_id:
            # Generate from genes and timestamp
            content = f"{self.creation_time.isoformat()}:{len(self.genes)}"
            self.dna_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def add_gene(self, gene: Gene):
        """Add a gene, replacing any existing gene of same type and name if dominant"""
        # Remove existing gene of same type and name if it's not dominant
        for i, g in enumerate(self.genes):
            if g.gene_type == gene.gene_type and g.name == gene.name:
                if gene.dominance == GeneDominance.DOMINANT:
                    self.genes[i] = gene
                elif g.dominance != GeneDominance.DOMINANT:
                    # Recessive vs recessive, keep highest confidence
                    if gene.confidence > g.confidence:
                        self.genes[i] = gene
                return
        self.genes.append(gene)
    
    def get_gene(self, gene_type: GeneType, name: Optional[str] = None) -> Optional[Gene]:
        """Retrieve a gene by type and optional name"""
        for g in self.genes:
            if g.gene_type == gene_type:
                if name is None or g.name == name:
                    return g
        return None
    
    def get_all_by_type(self, gene_type: GeneType) -> List[Gene]:
        return [g for g in self.genes if g.gene_type == gene_type]
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2) -> 'CognitiveDNA':
        """Create a mutated copy of this DNA"""
        new_dna = CognitiveDNA(
            dna_id="",
            genes=[],
            creation_time=datetime.now(),
            generation=self.generation + 1,
            parent_ids=[self.dna_id]
        )
        # Mutate each gene with probability mutation_rate
        for gene in self.genes:
            if random.random() < mutation_rate:
                new_dna.add_gene(gene.mutate(mutation_strength=mutation_strength))
            else:
                new_dna.add_gene(gene)
        return new_dna
    
    def crossover(self, other: 'CognitiveDNA') -> 'CognitiveDNA':
        """
        Create a new DNA by combining genes from two parents.
        Uses uniform crossover: each gene is randomly taken from one parent.
        If both parents have genes of same type/name, they are combined.
        """
        child_dna = CognitiveDNA(
            dna_id="",
            genes=[],
            creation_time=datetime.now(),
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.dna_id, other.dna_id]
        )
        
        # First, collect all gene types from both parents
        all_types = set()
        for g in self.genes + other.genes:
            all_types.add((g.gene_type, g.name))
        
        for gene_type, name in all_types:
            g1 = self.get_gene(gene_type, name)
            g2 = other.get_gene(gene_type, name)
            
            if g1 and g2:
                # Both have it: combine or choose one
                if random.random() < 0.5:
                    child_dna.add_gene(g1.combine(g2))
                else:
                    child_dna.add_gene(random.choice([g1, g2]))
            elif g1:
                child_dna.add_gene(g1)
            elif g2:
                child_dna.add_gene(g2)
        
        return child_dna
    
    def express(self) -> Dict[str, Any]:
        """
        Express the DNA into a concrete configuration dictionary.
        Handles dominance and epistasis: dominant genes override recessive ones,
        and some genes modify others.
        """
        # First, collect all genes by type
        genes_by_type: Dict[GeneType, List[Gene]] = {}
        for g in self.genes:
            genes_by_type.setdefault(g.gene_type, []).append(g)
        
        # Apply dominance: for each type, only the most dominant gene(s) survive
        expressed = {}
        for gtype, genes in genes_by_type.items():
            # Separate by dominance
            dominant = [g for g in genes if g.dominance == GeneDominance.DOMINANT]
            recessive = [g for g in genes if g.dominance == GeneDominance.RECESSIVE]
            
            if dominant:
                # Dominant genes take precedence; if multiple dominant, choose highest confidence
                selected = max(dominant, key=lambda g: g.confidence)
                expressed[gtype] = selected.value
            elif recessive:
                # Only recessive: choose highest confidence
                selected = max(recessive, key=lambda g: g.confidence)
                expressed[gtype] = selected.value
            else:
                # Codominant: blend or combine
                # For simplicity, just take the one with highest confidence
                selected = max(genes, key=lambda g: g.confidence)
                expressed[gtype] = selected.value
        
        # Apply epistatic modifications (genes that modify other genes)
        for g in self.genes:
            if g.dominance == GeneDominance.EPISTATIC and g.gene_type in expressed:
                # Modify the expressed value
                current = expressed[g.gene_type]
                if isinstance(current, (int, float)):
                    if g.name == "scale":
                        expressed[g.gene_type] = current * g.value
                    elif g.name == "add":
                        expressed[g.gene_type] = current + g.value
                # Add more modifications as needed
        
        return expressed
    
    def to_dict(self) -> Dict:
        return {
            "dna_id": self.dna_id,
            "genes": [g.to_dict() for g in self.genes],
            "creation_time": self.creation_time.isoformat(),
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CognitiveDNA':
        genes = [Gene.from_dict(g) for g in data["genes"]]
        return cls(
            dna_id=data["dna_id"],
            genes=genes,
            creation_time=datetime.fromisoformat(data["creation_time"]),
            fitness=data["fitness"],
            generation=data["generation"],
            parent_ids=data.get("parent_ids", []),
            metadata=data.get("metadata", {})
        )
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'CognitiveDNA':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

# ============= EVOLUTION ENGINE =============

class EvolutionEngine:
    """
    Manages a population of CognitiveDNAs and evolves them over generations.
    
    This is the core of the self-improvement loop: it selects the best DNAs,
    creates offspring through crossover and mutation, and maintains diversity.
    """
    
    def __init__(self, 
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism: int = 2,
                 selection_pressure: float = 1.5):
        """
        Args:
            population_size: Number of DNAs in the population
            mutation_rate: Probability of mutating a gene when creating offspring
            crossover_rate: Probability of using crossover vs mutation
            elitism: Number of top DNAs to carry over unchanged
            selection_pressure: Higher = more aggressive selection of best
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.selection_pressure = selection_pressure
        self.population: List[CognitiveDNA] = []
        self.generation = 0
        
    def initialize(self, seed_genes: List[Gene] = None):
        """Initialize population with random genes or provided seeds"""
        self.population = []
        for i in range(self.population_size):
            if seed_genes and i < len(seed_genes):
                # Start with seed genes, then mutate to create variation
                dna = CognitiveDNA(genes=[g for g in seed_genes])
                # Mutate slightly
                dna = dna.mutate(mutation_rate=0.2)
            else:
                # Random DNA from scratch (in practice, you'd have a gene pool)
                dna = self._create_random_dna()
            self.population.append(dna)
        self.generation = 0
    
    def _create_random_dna(self) -> CognitiveDNA:
        """Create a random DNA for initialization"""
        # This would be populated with random genes from a predefined set
        # For now, return empty
        return CognitiveDNA(genes=[])
    
    def evaluate_fitness(self, fitness_function: Callable[[CognitiveDNA], float]):
        """
        Evaluate fitness of all DNAs using provided function.
        The function should take a DNA and return a fitness score.
        """
        for dna in self.population:
            dna.fitness = fitness_function(dna)
    
    def select_parent(self) -> CognitiveDNA:
        """
        Select a parent using tournament selection with pressure.
        Higher fitness -> higher chance.
        """
        # Tournament: pick k random, choose best
        tournament_size = max(2, int(self.population_size * 0.2))
        tournament = random.sample(self.population, tournament_size)
        # Weight by fitness^selection_pressure
        fitnesses = [d.fitness ** self.selection_pressure for d in tournament]
        total = sum(fitnesses)
        if total == 0:
            return random.choice(tournament)
        pick = random.uniform(0, total)
        cum = 0
        for i, f in enumerate(fitnesses):
            cum += f
            if cum >= pick:
                return tournament[i]
        return tournament[-1]
    
    def evolve_one_generation(self):
        """
        Perform one generation of evolution.
        Returns the best DNA after evolution.
        """
        # Sort by fitness
        self.population.sort(key=lambda d: d.fitness, reverse=True)
        new_population = []
        
        # Elitism: keep top performers
        for i in range(min(self.elitism, len(self.population))):
            new_population.append(self.population[i])
        
        # Create offspring until population size is reached
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            if random.random() < self.crossover_rate:
                parent2 = self.select_parent()
                child = parent1.crossover(parent2)
            else:
                child = parent1
            
            # Apply mutation
            if random.random() < self.mutation_rate:
                child = child.mutate()
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        return self.get_best()
    
    def get_best(self) -> Optional[CognitiveDNA]:
        """Return the DNA with highest fitness"""
        if not self.population:
            return None
        return max(self.population, key=lambda d: d.fitness)
    
    def get_diversity(self) -> float:
        """Measure population diversity (average pairwise gene distance)"""
        if len(self.population) < 2:
            return 0.0
        distances = []
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                dist = self._gene_distance(self.population[i], self.population[j])
                distances.append(dist)
        return np.mean(distances) if distances else 0.0
    
    def _gene_distance(self, dna1: CognitiveDNA, dna2: CognitiveDNA) -> float:
        """Simple distance: number of different genes of same type"""
        # Get sets of (gene_type, name) for both
        set1 = {(g.gene_type, g.name) for g in dna1.genes}
        set2 = {(g.gene_type, g.name) for g in dna2.genes}
        union = set1.union(set2)
        intersection = set1.intersection(set2)
        if not union:
            return 0.0
        return 1 - len(intersection) / len(union)
    
    def save_population(self, directory: Path):
        """Save all DNAs in population"""
        directory.mkdir(parents=True, exist_ok=True)
        for i, dna in enumerate(self.population):
            filename = directory / f"dna_{self.generation}_{dna.dna_id}.json"
            dna.save(filename)
    
    def load_population(self, directory: Path):
        """Load DNAs from directory"""
        self.population = []
        for file in directory.glob("dna_*.json"):
            try:
                dna = CognitiveDNA.load(file)
                self.population.append(dna)
            except:
                continue
        # Extract generation from filenames (simplified)
        if self.population:
            # Try to infer generation
            self.generation = max(dna.generation for dna in self.population)

