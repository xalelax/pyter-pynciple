from mesa import Agent, Model
from scipy.stats import truncnorm

# Default parameters to reflect the choices in the original paper
default_age_dist = truncnorm(loc=25, scale=5, a=-1.4, b=7)
default_competency_dist = truncnorm(loc=7, scale=2, a=-6, b=3)


class Employee(Agent):
    """An agent which represents an employee"""

    def get_older(self):
        """Existential crisis incoming while writing this :-("""
        self.age += 1

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Random initialization
        self.age = self.model.age_distribution.rvs()
        self.competency = self.model.competency_distribution.rvs()

    def step(self):
        self.get_older()


class Company(Model):
    """
    Model of a company; pyramidal structure with n different levels;
    workers at different levels weigh differently on the efficiency of the
    company as a whole.

    In the paper, the default parameters are
        level_sizes = [1, 5, 11, 21, 41, 81] (for a total of 160 employees)
        level_weights = [1.0, 0.9, 0.8, 0.6, 0.4, 0.2]
    """

    def __init__(self,
                 level_sizes=[1, 5, 11, 21, 41, 81],
                 level_weights=[1.0, 0.9, 0.8, 0.6, 0.4, 0.2],
                 age_distribution=default_age_dist,
                 competency_distribution=default_competency_dist,
                 dismissal_threshold=4,
                 retirement_age=60,
                 competence_mechanism='common_sense',
                 promotion_strategy='best'):

        # Not the best way to pack all the constants I have
        self.level_sizes = level_sizes
        self.level_weights = level_weights
        self.age_distribution = age_distribution
        self.competency_distribution = competency_distribution
        self.dismissal_threshold = dismissal_threshold
        self.retirement_age = retirement_age
        self.competence_mechanism = competence_mechanism
        self.promotion_strategy = promotion_strategy

        assert (len(level_sizes) == len(level_weights)
                ), "Incompatible dimensions (level sizes and weights)"

        # Create agents
        self.levels = []
        for level_size in level_sizes:
            level = [Employee(self.next_id(), self) for j in range(level_size)]
            self.levels.append(level)
