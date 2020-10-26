from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from scipy.stats import truncnorm, binom, bernoulli
from numpy import dot

# Default parameters to reflect the choices in the original paper
default_age_dist = truncnorm(loc=25, scale=5, a=-1.4, b=7)
default_competency_dist = truncnorm(loc=7, scale=2, a=-3, b=1.5)


def calculate_efficiency(model):
    # TODO: for now I am assuming that max competency = 10; relax this later
    maximum_competency = 10
    max_outcome = maximum_competency * \
        dot(model.level_sizes, model.level_weights)

    outcome_per_level = [sum(employee.competency for employee in level)
                         for level in model.levels]
    total_outcome = dot(outcome_per_level, model.level_weights)

    return total_outcome/max_outcome


class Employee(Agent):
    """An agent which represents an employee"""

    def get_older(self):
        """Existential crisis incoming while writing this :-("""
        self.age += self.model.timestep_years

    @property
    def has_to_go(self):
        below_threshold = self.competency <= self.model.dismissal_threshold
        reached_retirement = self.age >= self.model.retirement_age
        leaving_randomly = self.model.dist_employees_leaving.rvs()
        return below_threshold or reached_retirement or leaving_randomly

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
                 level_sizes=(1, 5, 10, 30),
                 level_weights=(1.0, 0.8, 0.5, 0.2),
                 age_distribution=default_age_dist,
                 competency_distribution=default_competency_dist,
                 dismissal_threshold=4,
                 retirement_age=65,
                 timestep_years=1/12.,
                 initial_vacancy_fraction=0.2,
                 p_employee_leaves=1/24.,
                 competency_mechanism='common_sense',
                 promotion_strategy='best'):

        assert len(level_sizes) == len(level_weights), \
            "Incompatible dimensions (level sizes and weights)"

        assert hasattr(age_distribution, 'rvs'), \
            "age_distribution must have a rvs method returning random values"

        assert hasattr(competency_distribution, 'rvs'), \
            "competency_distribution must have a rvs method returning random values"

        assert promotion_strategy in ['best', 'worst', 'random'], \
            "Unrecognized promotion_strategy"

        assert competency_mechanism in ['common_sense', 'peter'], \
            "Unrecognized competency_mechanism"

        super().__init__()

        # Not the best way to pack all the constants I have
        self.level_sizes = level_sizes
        self.level_weights = level_weights
        self.age_distribution = age_distribution
        self.competency_distribution = competency_distribution
        self.dismissal_threshold = dismissal_threshold
        self.retirement_age = retirement_age
        self.timestep_years = timestep_years
        self.initial_vacancy_fraction = initial_vacancy_fraction
        self.dist_employees_leaving = bernoulli(p=p_employee_leaves)
        self.competency_mechanism = competency_mechanism
        self.promotion_strategy = promotion_strategy

        self.current_id = 0
        self.schedule = SimultaneousActivation(self)

        # Create agents
        self.levels = []
        for level_size in level_sizes:
            level = []
            n_initial_agents = binom(n=level_size, p=1-self.initial_vacancy_fraction).rvs()
            n_initial_agents = max(1, n_initial_agents)
            for i in range(n_initial_agents):
                agent = Employee(self.next_id(), self)
                self.schedule.add(agent)
                level.append(agent)
            self.levels.append(level)

        # Initialize data collection
        self.data_collector = DataCollector(
            model_reporters={'efficiency': calculate_efficiency})

    def remove_employees(self):
        for level in self.levels:
            for employee in level:
                if employee.has_to_go:
                    level.remove(employee)
                    self.schedule.remove(employee)

    def pick_for_promotion_from(self, source_level):
        if self.promotion_strategy == 'best':
            return max(source_level, key=lambda e: e.competency)
        elif self.promotion_strategy == 'worst':
            return min(source_level, key=lambda e: e.competency)
        elif self.promotion_strategy == 'random':
            return self.random.choice(source_level)

    def recalculate_competency(self, employee):
        # Peter hypothesis: competency in new role not dependent on competency
        #                   in previous role
        if self.competency_mechanism == 'peter':
            employee.competency = self.competency_distribution.rvs()
        # Common-sense: competency mostly transferable from previous role
        elif self.competency_mechanism == 'common_sense':
            # TODO: abstract parameters used here
            random_variation = self.random.uniform(-1, 1)
            employee.competency += random_variation
            employee.competency = min(max(0, employee.competency), 10)  # Clip

    def promote_employees(self):
        for upper_level, lower_level, target_size in zip(self.levels,
                                                         self.levels[1:],
                                                         self.level_sizes):
            while len(upper_level) < target_size and len(lower_level):
                chosen_employee = self.pick_for_promotion_from(lower_level)
                lower_level.remove(chosen_employee)
                self.recalculate_competency(chosen_employee)
                upper_level.append(chosen_employee)

    def hire_employees(self):
        bottom_level = self.levels[-1]
        bottom_level_target_size = self.level_sizes[-1]
        vacant_bottom_positions = bottom_level_target_size - len(bottom_level)

        for i in range(vacant_bottom_positions):
            agent = Employee(self.next_id(), self)
            self.schedule.add(agent)
            bottom_level.append(agent)

    def step(self):
        self.data_collector.collect(self)
        self.schedule.step()
        self.remove_employees()
        self.promote_employees()
        self.hire_employees()
