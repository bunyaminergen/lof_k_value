import numpy as np
import pandas as pd

from seaborn import load_dataset
from sklearn.neighbors import LocalOutlierFactor

data = load_dataset("iris")

data.head()

data.shape

def convert(species):
    if species == "setosa":
        return 0
    elif species == "versicolor":
        return 1
    elif species == "virginica":
        return 2
    else:
        raise ValueError("There is more than 3")

data["species"] = data["species"].apply(convert)


class LOF:

    def LOF(data        : pd.DataFrame,
            n_neighbors : list and int,
            visualize   : bool = True,
            tv          : int  = 20,
            nn          : bool = False,
            inplace     : bool = False
            ) -> pd.DataFrame:


        """

        By optimizing the n_neighbors hyperparameter of the Local Outlier Factor,
        it determines all the scores of certain values in the setting a range and
        returns the observations that are the intersection set of these scores, ie the strongest outliers.

        Note: I explain all the steps in case you wish to modify the function.


        Args:

            data: pd.DataFrame,
            dataframe

            n_neighbors: int, list or range
            For better results, it is not recommended to enter the minimum value as 2 and the maximum value as the number of observations.

            visualize: bool, False or True, (default False)
            To visualize LOF scores
            Note: The visualize argument has not been added yet. In process.

            tv: int, (default 20)
            tv (threshold value) It is to determine how many of the values (furthest from 1, outliers) are to be setting. Default is 20.
            Please consider that as you raise tv argument, the number of outliers may increase and inliers will also be return as outliers.

            nn: bool, (default False)
            nn (the nearest neighbors) argument can be use to assign the resulting outliers to the nearest neighbor. Default is False.
            Note: The nn argument has not been added yet. In process.

            inplace: bool, (default False)
            inplace argument is use to apply the returned results to the original dataset. Default is False.
            Note: The inplace argument has not been added yet. In process.


        Returns:

            Pandas DataFrame

        Examples:

            LOF.LOF(data, np.arange(10, 140, 10))

            or

            LOF.LOF(data, [10,20,30,40,50,60,70,80,90,100,110,120])


        """

        outl_list = list()
        strg_outl = list()

        # Creating the LOF function and apply the n_neighbors parameters with a for loop
        for i in n_neighbors:

            lof = LocalOutlierFactor(n_neighbors = i)
            lof.fit_predict(data)

            # after get the LOF scores, multiply by - and make them positive.
            lof_scores = -lof.negative_outlier_factor_

            # print(lof_scores)

            # Adding LOF scores as a new column
            data["LOF_" + str(i)] = lof_scores.tolist()

            LOF_columns = data["LOF_" + str(i)]

            # print(np.sort(LOF_columns)[-tv:])
            # print("_" * 50)

            # determine the highest LOF scores as much as the number observation entered into the TV.
            tv_ = np.sort(LOF_columns)[-tv]

            # print(tv_)
            # print("LOF_" + str(i))

            # assign the results to a variable, then append the index numbers of the results to the outl_list variable as a set.
            outliers_ = data[LOF_columns > tv_]
            outl_list.append(set(outliers_.index))

        print("Outliler index numbers for each n_neighbors value")
        print(*[sorted(i) for i in outl_list], sep="\n")
        print("_"*50)

        # get the intersection set of the variable outl_list (sets of otliers list) with all the results

        outl_inters = set.intersection(*outl_list)

        print("intersection set of n_neighbors results")
        print(sorted(list(outl_inters)))
        print("_"*50)

        # append the results to strg_outl list with the for loop and return as a dataframe.
        for i in sorted(list(outl_inters)):
            strg_outl.append(data.iloc[i])

        return pd.DataFrame(strg_outl)


LOF.LOF(data, np.arange(10, 150, 10))

# or

LOF.LOF(data, [10,20,40,70,110])
