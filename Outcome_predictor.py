import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
def getPlayersInfo(df,df1):
    df['TSR'] = df['total_strikes_succ'] / df['total_strikes_att']
    df['SSR'] = df['sig_strikes_succ'] / df['sig_strikes_att']
    df['TKR'] = df['takedowns_succ'] / df['takedowns_att']

    # Dropping tables which are not useful

    df = df.drop(['takedowns_succ', 'takedowns_att', 'total_strikes_succ',
                  'total_strikes_att', 'sig_strikes_succ', 'sig_strikes_att'], axis=1)

    # Merging again for future slicing. Storing as copy with name and id as extra columns
    df9 = pd.merge(df, df1[['fighter_id', 'first_name', 'nickname', 'last_name', 'reach_cm', 'wins', 'loss']],
                   on='fighter_id', how='left')

    # Rearranging the columns in df9
    df9 = df9[['first_name', 'nickname', 'last_name', 'fighter_id', 'TSR', 'SSR', 'TKR', 'knockdowns', 'submiss_att',
               'reversals', 'reach_cm', 'wins', 'loss', 'ctrl']]

    df9['nickname'] = df9['nickname'].astype(object)

    return df9
def datacleaning(df, df1, df2):
    '''
    Calculating the following:
    TSR: Total strike succeeded ratio
    SSR: Significant strike succeeded ratio
    TKR: Total takedowns succeeded ratio
    '''

    df['TSR'] = df['total_strikes_succ'] / df['total_strikes_att']
    df['SSR'] = df['sig_strikes_succ'] / df['sig_strikes_att']
    df['TKR'] = df['takedowns_succ'] / df['takedowns_att']

    # Dropping tables which are not useful

    df = df.drop(['takedowns_succ', 'takedowns_att', 'total_strikes_succ',
                  'total_strikes_att', 'sig_strikes_succ', 'sig_strikes_att'], axis=1)

    # Merging stats df and fighter df1 using left join
    df8 = pd.merge(df, df1[['fighter_id', 'reach_cm', 'wins', 'loss']], on='fighter_id', how='left')

    # In[10]:

    # Merging again for future slicing. Storing as copy with name and id as extra columns
    df9 = pd.merge(df, df1[['fighter_id', 'first_name', 'nickname', 'last_name', 'reach_cm', 'wins', 'loss']],
                   on='fighter_id', how='left')

    # Rearranging the columns in df9
    df9 = df9[['first_name', 'nickname', 'last_name', 'fighter_id', 'TSR', 'SSR', 'TKR', 'knockdowns', 'submiss_att',
               'reversals', 'reach_cm', 'wins', 'loss', 'ctrl']]

    df9['nickname'] = df9['nickname'].astype(object)

    median_reach = df8['reach_cm'].median()

    df8['reach_cm'] = df8['reach_cm'].fillna(median_reach)

    # Filling null values with 0
    df8['TSR'] = df8['TSR'].fillna(0)
    df8['SSR'] = df8['SSR'].fillna(0)
    df8['TKR'] = df8['TKR'].fillna(0)

    df8 = df8[['fighter_id', 'TSR', 'SSR', 'TKR', 'knockdowns', 'submiss_att', 'reversals', 'reach_cm', 'wins', 'loss',
               'ctrl']]

    # Checking missing values
    missing99 = df2.isna().sum()

    # Dropping unnecessary coluns
    df2 = df2.drop(['num_rounds', 'title_fight', 'weight_class', 'gender', 'result_details', 'result',
                    'finish_round', 'finish_time', 'fight_url', 'event_id'], axis=1)

    # Merging fight df2 with stat df8 on f_1 using left join
    df2 = pd.merge(df2, df8[
        ['fighter_id', 'submiss_att', 'reversals', 'ctrl', 'knockdowns', 'TSR', 'SSR', 'TKR', 'reach_cm', 'wins',
         'loss']], left_on=['f_1'], right_on=['fighter_id'], how='left')

    # Renaming the names of TSR as f_1
    df2 = df2.rename(columns={'TSR': 'f1_TSR', 'SSR': 'f1_SSR', 'TKR': 'f1_TKR', 'knockdowns': 'f1_knockdowns',
                              'submiss_att': 'f1_sub', 'reversals': 'f1_reversals', 'ctrl': 'f1_ctrl',
                              'reach_cm': 'f1_reach', 'wins': 'f1_wins', 'loss': 'f1_loss'})

    # Merging fight df2 with stat df8 on f_2 using left join
    df2 = pd.merge(df2, df8[
        ['fighter_id', 'submiss_att', 'reversals', 'ctrl', 'knockdowns', 'TSR', 'SSR', 'TKR', 'reach_cm', 'wins',
         'loss']], left_on=['f_2'], right_on=['fighter_id'], how='left')

    # Renaming the names of TSR as f_2
    df2 = df2.rename(columns={'TSR': 'f2_TSR', 'SSR': 'f2_SSR', 'TKR': 'f2_TKR', 'knockdowns': 'f2_knockdowns',
                              'submiss_att': 'f2_sub', 'reversals': 'f2_reversals', 'ctrl': 'f2_ctrl',
                              'reach_cm': 'f2_reach', 'wins': 'f2_wins', 'loss': 'f2_loss'})
    # Dropping duplicate and unnecessary columns
    df2 = df2.drop(['fighter_id_x', 'fighter_id_y', 'referee'], axis=1)

    # Drop rows with missing values
    df2 = df2.dropna()

    # Coputing the difference between stats of both the fighters
    df2['TSR'] = df2['f1_TSR'] - df2['f2_TSR']
    df2['SSR'] = df2['f1_SSR'] - df2['f2_SSR']
    df2['TKR'] = df2['f1_TKR'] - df2['f2_TKR']
    df2['knockdowns'] = df2['f1_knockdowns'] - df2['f2_knockdowns']
    df2['sub'] = df2['f1_sub'] - df2['f2_sub']
    df2['reversals'] = df2['f1_reversals'] - df2['f2_reversals']
    df2['reach'] = df2['f1_reach'] - df2['f2_reach']
    df2['wins'] = df2['f1_wins'] - df2['f2_wins']
    df2['loss'] = df2['f1_loss'] - df2['f2_loss']

    # In[39]:

    df2['ctrl'] = df2['f1_ctrl'] - df2['f2_ctrl']

    # Dropping individual stats
    df2 = df2.drop(['f1_sub', 'f1_reversals', 'f1_ctrl',
                    'f1_knockdowns', 'f1_TSR', 'f1_SSR', 'f1_TKR', 'f1_reach', 'f1_wins',
                    'f1_loss', 'f2_sub', 'f2_reversals', 'f2_ctrl', 'f2_knockdowns',
                    'f2_TSR', 'f2_SSR', 'f2_TKR', 'f2_reach', 'f2_wins', 'f2_loss'], axis=1)

    # Rearranging columns
    df2 = df2[['fight_id', 'f_1', 'f_2', 'TSR', 'SSR', 'TKR', 'knockdowns',
               'sub', 'reversals', 'reach', 'wins', 'loss', 'ctrl', 'winner']]

    # Assigning binary integers to winners
    # fighter 1: 1
    # fighter 2: 0
    df2['winner'] = (df2['f_1'] == df2['winner']).astype(int)

    df2 = df2.drop(['fight_id', 'f_1', 'f_2'], axis=1)

    return df2

    # Drop fighter ids for analysis




def train(df2,row_as_list,row_as_list1):

    new_list = row_as_list[4:]
    new_list1 = row_as_list1[4:]

    # Target variable is the last column
    X = df2.iloc[:, :-1].values
    y = df2.iloc[:, -1].values

    # In[50]:

    # Splitting training and testing data - 75-25

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Running the model
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))

    difference = [a - b for a, b in zip(new_list, new_list1)]
    print(difference)

    # In[72]:

    result = classifier.predict(sc.transform([difference]))

    # In[73]:
    resultdtr = ""
    if result == 1:
        resultdtr = row_as_list[0] + " '" + row_as_list[1] + "' " + row_as_list[2] + " wins"
    else:
        resultdtr = row_as_list1[0] + " '" + row_as_list1[1] + "' " + row_as_list1[2] + " wins"

    # In[74]:

    # Calculate the win probabilities using predict_proba
    probabilities = classifier.predict_proba(sc.transform([difference]))

    perc1 = probabilities[0][1] * 100
    perc2 = probabilities[0][0] * 100

    # probabilities[0][0] is the probability of the first class (e.g., Fighter 1 losing)
    # probabilities[0][1] is the probability of the second class (e.g., Fighter 1 winning)

    # Display the win probabilities
    print(f"Win Probability for {row_as_list[0]} {row_as_list[1]} {row_as_list[2]}: {probabilities[0][1] * 100:.2f}%")
    print(f"Win Probability for {row_as_list1[0]} { row_as_list1[1]} { row_as_list[2]}: {probabilities[0][0] * 100:.2f}%")
    return resultdtr, perc1, perc2




def getFighterInfo(fighterId,df9):

    row = df9[df9['fighter_id'] == fighterId]

    # Check if the row exists
    if not row.empty:
        # Convert the row to a list
        row_as_list = row.values.tolist()[0]
        # Store it in a variable
        fighter_data = row_as_list
        print(f"Data for fighter {fighterId}: {fighter_data}")
    else:
        print(f"No data found for fighter ID {fighterId}")

    new_list = row_as_list[4:]
    info_list = row_as_list[0:3]
    f_name, n_name, l_name = row_as_list[0], row_as_list[1], row_as_list[2]
    return row_as_list



def getFighterNames():
    df = pd.read_csv("/Users/rohannair/Projects/UFC/Octagon Oracle - Copy/new_stats.csv")
    df1 = pd.read_csv("/Users/rohannair/Projects/UFC/Octagon Oracle - Copy/UFC_FIGHTER.csv")
    df9 = getPlayersInfo(df, df1)
    name_list = []
    for index,row in df9.iterrows():
        fname = row["first_name"]
        nName = row["nickname"]
        lName = row["last_name"]

        if nName is np.NaN:
            nName = ""

        name = fname+" '"+nName+"' "+lName
        name_list.append(name)

    return name_list

def getFighterId(fighterNameUser):
    df = pd.read_csv("/Users/rohannair/Projects/UFC/Octagon Oracle - Copy/new_stats.csv")
    df1 = pd.read_csv("/Users/rohannair/Projects/UFC/Octagon Oracle - Copy/UFC_FIGHTER.csv")
    df9 = getPlayersInfo(df, df1)

    name_map = {}
    for index, row in df9.iterrows():
        fname = row["first_name"]
        nName = row["nickname"]
        lName = row["last_name"]
        fighterId = row["fighter_id"]

        if pd.isna(nName):  # Checking for NaN in a more reliable way
            nName = ""

        name = f"{fname} '{nName}' {lName}".strip()  # .strip() to remove any leading/trailing spaces

        name_map[fighterId] = name  # Using fighterId as key and name as value

    id = list(name_map.keys())[list(name_map.values()).index(fighterNameUser)]

    return id





def prdict_outcome(fighter1,fighter2):
    # Load stats data in df
    df = pd.read_csv("/Users/rohannair/Projects/UFC/Octagon Oracle - Copy/new_stats.csv")
    df1 = pd.read_csv("/Users/rohannair/Projects/UFC/Octagon Oracle - Copy/UFC_FIGHTER.csv")
    df2 = pd.read_csv("/Users/rohannair/Projects/UFC/Octagon Oracle - Copy/UFC_FIGHT.csv")

    df10 = datacleaning(df,df1,df2)
    df9 = getPlayersInfo(df,df1)

    player1Id = getFighterId(fighter1)
    player2Id = getFighterId(fighter2)

    row_as_list = getFighterInfo(player1Id,df9)
    row_as_list1 = getFighterInfo(player2Id,df9)


    traindf, prob1, prob2 = train(df10, row_as_list, row_as_list1)
    return traindf,prob1,prob2

