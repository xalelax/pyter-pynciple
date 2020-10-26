from model import Company
from itertools import product
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import click

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def simulate_sample_company(n_steps, **kwargs):
    """
    Runs a model for n_steps and returns a pandas.DataFrame
    containing the data collected at each step.
    kwargs are forwarded to init of company.
    """
    model = Company(**kwargs)
    for i in range(n_steps):
        model.step()
    df = model.data_collector.get_model_vars_dataframe().copy()
    df['i'] = df.index
    df['competency_mechanism'] = model.competency_mechanism
    df['promotion_strategy'] = model.promotion_strategy
    df['year'] = df['i'] * model.timestep_years
    return df


def plot_results(ensemble_df, save_figure=None):
    eff_v_time = sns.relplot(col='competency_mechanism',
                             hue='promotion_strategy',
                             x='year',
                             y='efficiency',
                             data=ensemble_df,
                             kind='line',
                             legend=None)

    eff_v_time.set(xscale='log', xlim=(1, None))
    eff_v_time.set_axis_labels('Time (years)', 'Efficiency [%]')

    axes = eff_v_time.axes.flatten()
    axes[0].legend(['Promote Best', 'Promote Worst', 'Promote Random'],
                   loc='upper left',
                   title='Promotion strategy')
    axes[0].set_title("Common sense hypothesis")
    axes[1].set_title("Peter hypothesis")

    if save_figure:
        eff_v_time.savefig('eff_v_time.svg')
        eff_v_time.savefig('eff_v_time.png')
    plt.show()


def run_parallel(parameter_combinations, n_steps, n_proc=None):
    if not n_proc:
        n_proc = mp.cpu_count()

    pbar = tqdm(total=len(parameter_combinations))
    with mp.Pool(n_proc) as pool:
        results = []
        for m, s, _ in parameter_combinations:
            result = pool.apply_async(simulate_sample_company,
                                      kwds={'competency_mechanism': m,
                                            'promotion_strategy': s,
                                            'n_steps': n_steps},
                                      callback=lambda _: pbar.update())
            results.append(result)
        pool.close()
        pool.join()

    return map(lambda x: x.get(), results)


@click.command()
@click.option('-n', '--n-runs', default=1)
@click.option('-s', '--n-steps', default=200)
@click.option('--savefig/--no-savefig', default=False)
def main(n_runs, savefig, n_steps):
    mechanisms = ['common_sense', 'peter']
    strategies = ['best', 'worst', 'random']
    parameter_combinations = list(
        product(mechanisms, strategies, range(n_runs)))

    results = run_parallel(parameter_combinations, n_steps)

    ensemble_df = pd.concat(results, ignore_index=True)

    plot_results(ensemble_df, savefig)


if __name__ == "__main__":
    main()