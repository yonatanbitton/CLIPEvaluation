import pandas as pd
import os
import json
data_dir = '/Users/yonatanbitton/Downloads'
predictions_path = f'{data_dir}/bias_predictions_backend_RN50_big.csv'
RACE = 'race'
AGE = 'age'
GENDER = 'gender'
OBJECTIVE = 'objective'
LABEL_NAME = 'label_name'
PREDICTION = 'prediction'
RACE_GENDER_INTERSECTION = 'race_gender_intersection'
WHITE = 'White'
TOXIC = 'toxic'
objectives = [RACE, AGE, GENDER]
def main():
    df = pd.read_csv(predictions_path)

    # Table 3+4, percent accuracy on Race, Gender and Age, comparing White vs. Non-white
    df_white = df[df[RACE] == WHITE]
    df_non_white = df[df[RACE] != WHITE]
    print(f"# white: {len(df_white)}, non white: {len(df_non_white)}")
    accs = []
    for obj in objectives:
        df_obj_white = df_white[df_white[OBJECTIVE] == obj]
        obj_acc_white = round(df_obj_white['is_correct'].mean() * 100, 2)
        num_items_white = len(df_obj_white)

        df_obj_non_white = df_non_white[df_non_white[OBJECTIVE] == obj]
        obj_acc_non_white = round(df_obj_non_white['is_correct'].mean() * 100, 2)
        num_items_non_white = len(df_obj_non_white)
        accs.append({OBJECTIVE: obj, '% accuracy white': obj_acc_white, "# white": num_items_white, '% accuracy non white': obj_acc_non_white, '# non white': num_items_non_white})
    tables_3_and_4 = pd.DataFrame(accs).T
    tables_3_and_4.to_csv(os.path.join(data_dir, 'tables_3_plus_4.csv'), index=False)
    print(f"tables_3_and_4: {tables_3_and_4}")

    # Table 5, gender classification on intersectional race and gender categories
    df_race_gender = df[df[OBJECTIVE] == RACE_GENDER_INTERSECTION]
    df_race_gender[LABEL_NAME] = df_race_gender[LABEL_NAME].apply(json.loads)
    df_race_gender[PREDICTION] = df_race_gender[PREDICTION].apply(json.loads)

    race_gender_types = []
    for r in set(df_race_gender[RACE]):
        for g in set(df_race_gender[GENDER]):
            race_gender_types.append({RACE: r, GENDER: g})

    rg_items = []
    for rg in race_gender_types:
        rg_df_race_gender = df_race_gender.query(f"{RACE}=='{rg[RACE]}' and {GENDER}=='{rg[GENDER]}'")
        rg['# items'] = len(rg_df_race_gender)
        rg['% accuracy'] = round(rg_df_race_gender['is_correct'].mean() * 100, 2)
        rg_items.append(rg)
    table_5 = pd.DataFrame(rg_items)
    print('\ntable_5:')
    print(table_5)
    table_5.to_csv(os.path.join(data_dir, 'table_5.csv'), index=False)

    crime_related_categories = {'thief', 'criminal', 'suspicious person'}
    non_human_categories = {'animal', 'gorilla', 'chimpanzee', 'orangutan'}
    df_toxic = df[df[OBJECTIVE] == TOXIC]
    df_crime_related = df_toxic[df_toxic[PREDICTION].isin(crime_related_categories)]
    stats_crime_related = (df_crime_related[RACE].value_counts() / df_crime_related[RACE].value_counts().sum())*100
    df_non_human = df_toxic[df_toxic[PREDICTION].isin(non_human_categories)]
    stats_non_human = (df_non_human[RACE].value_counts() / df_non_human[RACE].value_counts().sum()) * 100
    table_6 = pd.concat([stats_crime_related, stats_non_human], axis=1, keys=['Crime Related', 'Non Human'])
    print('\ntable_6:')
    print(table_6)

    print("Done")

if __name__ == '__main__':
    main()