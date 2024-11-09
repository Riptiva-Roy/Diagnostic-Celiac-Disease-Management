BloodLxst
bloodlust.x
Online
Cos Rats

BloodLxst — Today at 1:56 PM
Alright yeah I assumed  random forest model too for some reason
jessica — Today at 2:55 PM
Guess what!
BloodLxst — Today at 2:55 PM
HI
jessica — Today at 2:55 PM
Idk how accurate accurate our algorithm is but it works!
look
BloodLxst — Today at 2:55 PM
YAYYYY
Im almost done with th technical documentation
send code i need to write that up too
Wait its just PCOS right?
jessica — Today at 2:56 PM
The 0.9 is how accurate our model is

ignore the "please enter following details line"
Image
but we entered like an array of fake data and thats what it outputted
jessica — Today at 2:57 PM
yes
jessica — Today at 2:57 PM
ok
BloodLxst — Today at 2:57 PM
What was the data measuring?
jessica — Today at 2:57 PM
features like
Image
but thats all for the doctors to input
BloodLxst — Today at 2:58 PM
perfect
ok just send whatever u used
we have the content but presentation is impprtant too i wanna make sure we dont miss stuff hehe
if we have some time i can look into extending it too
jessica — Today at 2:59 PM
we might want to edit the code for example if the doctor doesn't have info like ADH(ml), then itll be omitted from the data set
BloodLxst — Today at 3:00 PM
thats ok
jessica — Today at 3:00 PM
bc right now its assuming the doctor has every single piece of info
BloodLxst — Today at 3:00 PM
thats fine
jessica — Today at 3:00 PM
do you still want the code?
BloodLxst — Today at 3:00 PM
yeah
our model isnt gonna be perfect we have assumptions and limitations thats fine as long as we specify too
jessica — Today at 3:00 PM
# Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
Expand
algorithm.py
4 KB
hold on some of the lines should be commented out
BloodLxst — Today at 3:01 PM
okk
jessica — Today at 3:01 PM
actually nvm its all good
also could u give us github access bc we want to figure out front and back end thing with the website
BloodLxst — Today at 3:02 PM
u have to send a push request
with the file
jessica — Today at 3:03 PM
ok hold on
BloodLxst — Today at 3:04 PM
I can put this file in myself
BloodLxst — Today at 3:04 PM
For this one go ahead
ill approve when it comes thru or u can tell me when
BloodLxst — Today at 3:24 PM
Guys are you down to integrate this with possibly a sort of health app focussed on menstrual health that kind of takes lifestyle/habit into account to pre-diagnose or indicate other hormonal issues
i.e. users can also input some things like 
steps, cravings (sometimes this indicates a lack of a certain nutrient if there is a trend), mood(swings/general trends) through a sort of mini quiz etc
like certain stuff like this does exist but if we can use it as a tool to help this im just wondering
Actually the best thing would be to integrate this onto the website
@Charmy can u send what u got so far
I can work on that too
jessica — Today at 3:39 PM
steps to connect front and backend
create a python server
create end point/api in server
That will call machine model (so put code in api)
2a. Then return result
Then front end calls api With fetch
We should use fetch (because we're using html and dont have package manager)
jessica — Today at 3:39 PM
yup she is rn
Charmy — Today at 3:40 PM
sorry no phone
do  uwanna collab with me on replit
@BloodLxst
https://replit.com/join/qarwioohwx-bp3669
replit
Sign Up
Build and deploy software collaboratively with the power of AI without spending a second on setup.
BloodLxst — Today at 3:41 PM
im in it
Charmy — Today at 3:41 PM
do you feel okay
do you wanna come to yeh
ok so we need to make different html files for every link we have
thats what im working on
rn
BloodLxst — Today at 3:44 PM
wait r u cool with changing it from ovu aid lol
didnt hear back so assumed itd be ok but js checking
Charmy — Today at 3:45 PM
ya go ahead
jessica — Today at 3:45 PM
Could u update this code
# Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
Expand
algorithm.py
4 KB
the previous one was just a test and this one actually has user input now
BloodLxst — Today at 3:48 PM
OK great
I was actually gonna code that otherwise lolll
alr so what is everyone doing rn
Charmy — Today at 3:49 PM
i think i mpretty much done with the website
BloodLxst — Today at 3:49 PM
might add some more functions to the code omnce im done w the pitch/logo rq
then add that to the docs ofc
Charmy — Today at 3:49 PM
iwe need to figure out how to connect the ml to html
BloodLxst — Today at 3:49 PM
Can someone look into submission procedure for dora too
jessica — Today at 3:49 PM
im trying to do step one in this 
jessica — Today at 3:50 PM
this is another thing to work on
BloodLxst — Today at 3:50 PM
r we good with this
Image
BloodLxst — Today at 3:50 PM
alright yeah
I'm gonna look into some added functionality once this basis is done
BloodLxst — Today at 3:51 PM
okkk perf
Charmy — Today at 3:52 PM
love
jessica — Today at 3:53 PM
is it ok if this runs only locally
for front and back end that would be easier and additionall in the ml, it's copying the file path locally
BloodLxst — Today at 3:54 PM
Sure
If we have time at the end lets try make it global
but otherwise how about we run it and maybe one of u could make like a screenrecording?
I think that'd be a good addition
jessica — Today at 3:59 PM
Aryah is working on that rn
BloodLxst — Today at 3:59 PM
I'm basically done with the pitch and got a good lil spiel going can deliever that
looking into some extra functions then great that u guys got those steps going
BloodLxst — Today at 4:00 PM
I'm keeping the slides minimal to prevent overload fbut got more content saved to talk abt
BloodLxst — Today at 4:17 PM
lol i was multitasking on diff things in the project for the last few hours at once not the best idea
anyway i updated some of the code
asterixA — Today at 4:17 PM
with what
BloodLxst — Today at 4:18 PM
lmao bruh
wait a sec
wait it'll take too long to explain nvm i'll just send the file
# Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
Expand
file.py
6 KB
basically refinements for accuracy like  cross validation and grid search
BloodLxst — Today at 4:27 PM
@Charmy lmk which files r finalised then ill add them to the git
asterixA — Today at 4:34 PM
I added the dropping table if values are empty
can you edit this file instead? @BloodLxst 
# Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
Expand
algorithm (1).py
5 KB
@BloodLxst
BloodLxst — Today at 4:36 PM
This is the final one for. now then
﻿
# Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# reading the dataset
file_path = r"C:\Users\ap9029\Desktop\Princeton\Data\PCOS Dataset.csv"
data = pd.read_csv(file_path)

data["AMH(ng/mL)"] = pd.to_numeric(data["AMH(ng/mL)"], errors='coerce')
data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors='coerce')

data['Marraige Status (Yrs)'] = data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median())
data['II    beta-HCG(mIU/mL)'] = data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median())
data['AMH(ng/mL)'] = data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].median())
data['Fast food (Y/N)'] = data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].mode()[0])

# Clearing up the extra space in the column names (optional)
data.columns = [col.strip() for col in data.columns]

# Identifying non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)

# Converting non-numeric columns to numeric where possible
for col in non_numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Dropping rows with any remaining non-numeric values
data.dropna(inplace=True)

# Preparing data for model training
columns_to_drop = ["PCOS (Y/N)", "Sl. No", "Patient File No.", "Marraige Status (Yrs)", "Blood Group", "II    beta-HCG(mIU/mL)", "TSH (mIU/L)", "Waist:Hip Ratio"]
X = data.drop(columns=columns_to_drop)
y = data["PCOS (Y/N)"]

# Splitting the data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fitting the RandomForestClassifier to the training set
rfc = RandomForestClassifier()
rfc.fit(X_train_scaled, y_train)

# Making prediction and checking the test set
pred_rfc = rfc.predict(X_test_scaled)
accuracy = accuracy_score(y_test, pred_rfc)
print(accuracy)

# Example: Collecting user input for the features
print("Please enter the following details:")

user_input = []
provided_columns = []

# Loop over the columns of X
for column in X.columns:
    # Prompt the user to enter the value for each feature
    input_value = input(f"Enter the value for {column}: ")
    if len(input_value) > 0:
        value = float(input_value)
        # Append the input value to the user_input list
        user_input.append(value)
        # Keep track of columns that have been provided
        provided_columns.append(column)

# Create a DataFrame with the provided columns
user_input_df = pd.DataFrame([user_input], columns=provided_columns)

# Ensure the user input DataFrame has the same columns as the training data
for col in X.columns:
    if col not in user_input_df.columns:
        user_input_df[col] = X_train[col].median()

# Reorder columns to match the training data
user_input_df = user_input_df[X.columns]

# Scale the user input
user_input_scaled = scaler.transform(user_input_df)

# Get the probability of PCOS
probabilities = rfc.predict_proba(user_input_scaled)

# Extract probability for PCOS (class 1)
probability_pcos = probabilities[0][1]  # Probability of PCOS (class 1)
probability_non_pcos = probabilities[0][0]  # Probability of non-PCOS (class 0)

# Output the result
print(f"Probability of PCOS: {probability_pcos * 100:.2f}%")
print(f"Probability of non-PCOS: {probability_non_pcos * 100:.2f}%")
algorithm (1).py
5 KB
