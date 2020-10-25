from model import Company
from itertools import product
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd

import seaborn as sns
sns.set_style("whitegrid")


def get_sample_company(n_steps=2000, **kwargs):
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
    return df


if __name__ == "__main__":

    mechanisms = ['common_sense', 'peter']
    strategies = ['best', 'worst', 'random']
    n_runs = 50
    n_proc = mp.cpu_count()

    parameter_combinations = list(
        product(mechanisms, strategies, range(n_runs)))

    pbar = tqdm(total=len(parameter_combinations))
    with mp.Pool(n_proc) as pool:
        results = []
        for m, s, _ in parameter_combinations:
            result = pool.apply_async(get_sample_company,
                                      kwds={'competency_mechanism': m,
                                            'promotion_strategy': s},
                                      callback=lambda _: pbar.update())
            results.append(result)

        pool.close()
        pool.join()

    results = map(lambda x: x.get(), results)
    ensemble_df = pd.concat(results, ignore_index=True)

    eff_v_time = sns.relplot(col='competency_mechanism',
                             hue='promotion_strategy',
                             x='i',
                             y='efficiency',
                             data=ensemble_df,
                             kind='line',
                             legend=None)

    eff_v_time.set(xscale='log', xlim=(1, None))
    eff_v_time.set_axis_labels('Simulation step', 'Efficiency [%]')

    axes = eff_v_time.axes.flatten()
    axes[0].legend(['Promote Best', 'Promote Worst', 'Promote Random'],
                   loc='upper left',
                   title='Promotion strategy')
    axes[0].set_title("Common sense hypothesis")
    axes[1].set_title("Peter principle")

    eff_v_time.savefig('eff_v_time.svg')
    eff_v_time.savefig('eff_v_time.png')
