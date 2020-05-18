from itertools import product
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd

from model import Company


def get_sample_company(n_steps=100, **kwargs):
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


mechanisms = ['common_sense', 'peter']
strategies = ['best', 'worst', 'random']
n_runs = 10
n_proc = mp.cpu_count()

parameter_combinations = list(product(mechanisms, strategies, range(n_runs)))

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
