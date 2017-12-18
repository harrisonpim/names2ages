import numpy as np
import pandas as pd
import datetime


def get_US_baby_names():
    '''
    loads the raw US baby name data stored in the data/raw/ directory

    Returns
    -------
    df : pd.DataFrame
        dataframe containing all US baby name data from 1880 - 2017
    '''
    df_dict = {year: pd.read_csv('./data/raw/yob{}.txt'.format(year),
                                 names=['Name', 'Sex', 'Count'])
               for year in range(1880, 2017)}

    for year in df_dict:
        df_dict[year]['Year'] = year

    return pd.concat([df_dict[i] for i in df_dict], axis=0)


def year_given_name(df):
    '''
    returns two dataframes (one male and one female) giving the probability of 
    being born with a given name in a given year.
    
    Parameters
    ----------
    df : pd.DataFrame
        columns = Name, Sex, Count, Year
        for all names in all available years

    Returns
    -------
    male_prob_year_given_name : pd.DataFrame
        columns = male names
        indexes = years
        values = probability of being born that year, given that name
    female_prob_year_given_name : pd.DataFrame
        columns = female names
        indexes = years
        values = probability of being born that year, given that name
    '''
    df_males   = df[df['Sex'] == 'M']
    df_females = df[df['Sex'] == 'F']

    male_pivot   = df_males.pivot_table(columns='Name', index='Year', values='Count')
    female_pivot = df_females.pivot_table(columns='Name', index='Year', values='Count')

    male_prob_name_given_year   = male_pivot.div(male_pivot.sum(axis=1), axis=0)
    female_prob_name_given_year = female_pivot.div(female_pivot.sum(axis=1), axis=0)

    male_prob_year_given_name   = male_pivot.div(male_pivot.sum(axis=0), axis=1)
    female_prob_year_given_name = female_pivot.div(female_pivot.sum(axis=0), axis=1)
    
    return male_prob_year_given_name, female_prob_year_given_name


def get_year_distribution(population_prob_year_given_name, subpopulation_names):
    '''
    get the distribution of probable birthyears for a given list of names

    Parameters
    ----------
    pop_prob_year_given_name : pd.DataFrame
        the probabilities of being born each year given a particular name
    subpop_names : numpy.ndarray
        a list of names from a subpopulation whose ages we want to infer

    Returns
    -------
    probabilities : pd.Series
        the inferred distribution (normalised to sum of 1) of birthyears in 
        the subpopulation 
    '''
    intersection = np.intersect1d(subpopulation_names, 
                                  population_prob_year_given_name.columns.values)

    sub_names_intersect = [n for n in subpopulation_names if n in intersection]
    sub_prob_name = pd.Series(sub_names_intersect).value_counts(normalize=True) 
    
    probabilities = {year: ((sub_prob_name.loc[intersection] * 
                             population_prob_year_given_name[intersection].loc[year])
                            .sum()) 
                     for year in range(1880, 2017)}
    
    return pd.Series(probabilities)


def age_from_birthdate(birthdate):
    '''
    takes a birthday and returns an approximate age in years

    Parameters
    ----------
    birthdate : datetime.datetime
        the candidate date of birth

    Returns
    -------
    age : int
        approximate number of years since given birthdate
    '''
    return datetime.datetime.today().year - birthdate.year