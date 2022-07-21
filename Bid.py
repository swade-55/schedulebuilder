import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Schdule Builder App
This app predicts optimal schedules for C&S Wholesale Grocers labor planning!
""")

st.sidebar.header('User Input Features')

#st.sidebar.markdown("""
#[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
#""")


def user_input_features():
    Sunday = st.sidebar.slider('Sunday Needs (heads)', 1,100,12)
    Monday = st.sidebar.slider('Monday Needs (heads)', 1,100,13)
    Tuesday = st.sidebar.slider('Tuesday Needs (heads)', 1,100,11)
    Wednesday = st.sidebar.slider('Wednesday Needs (heads)', 1,100,11)
    Thursday = st.sidebar.slider('Thursday Needs (heads)', 1,100,9)
    Friday = st.sidebar.slider('Friday Needs (heads)', 1,100,5)
    data = [Sunday,Monday,Tuesday,Wednesday,Thursday,Friday]
    return data
input_df = user_input_features()


import pandas as pd
from pulp import *
import matplotlib.pyplot as plt
from itertools import chain, repeat

def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(repeat(tuple(iterable), n))

# Staff needs per Day
n_staff = input_df
jours = ['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Staff
df_staff = pd.DataFrame({'Days': jours, 'Staff Demand':n_staff})
df_staff[['Days', 'Staff Demand']].plot.bar(x='Days', figsize = (30, 10), fill=True, color='black')
plt.title('Workforce Ressources Demand by Day',fontsize=16)
plt.xlabel('Day of the week',fontsize=16)
plt.ylabel('Number of Workers',fontsize=16)
plt.show()

# Create circular list of days
n_days = [i for i in range(6)]
n_days_c = list(ncycles(n_days, 2)) 

# Working days
list_in = [[n_days_c[j] for j in range(i , i + 4)] for i in n_days_c]

# Days off
list_excl = [[n_days_c[j] for j in range(i + 1, i + 3)] for i in n_days_c]

# The class has been initialize, and x, and days defined
model = LpProblem("Minimize Staffing", LpMinimize)

# Create Variables
start_jours = ['Shift: ' + i for i in jours]
x = LpVariable.dicts('shift_', n_days, lowBound=0, cat='Integer')

# Define Objective
model += lpSum([x[i] for i in n_days])

# Add constraints
for d, l_excl, staff in zip(n_days, list_excl, n_staff):
    model += lpSum([x[i] for i in n_days if i not in l_excl]) >= staff

# Solve Model
model.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[model.status])

# How many workers per day ?
dct_work = {}
dico_work = {}
for v in model.variables():
    dct_work[int(v.name[-1])] = int(v.varValue)
    dico_work[v.name] = int(v.varValue)
dico_work

# Show workers schedule
dict_sch = {}
for day in dct_work.keys():
    dict_sch[day] = [dct_work[day] if i in list_in[day] else 0 for i in n_days]
df_sch = pd.DataFrame(dict_sch).T
df_sch.columns = jours
df_sch.index = start_jours
# The optimized objective function value is printed to the screen
print("Total number of Staff = ", pulp.value(model.objective))

# Sum by day
df_sch.sum(axis =0)

df_supp = df_staff.copy().set_index('Days')
df_supp['Staff Supply'] = df_sch.sum(axis = 0)
df_supp['Extra_Ressources'] = df_supp['Staff Supply'] - df_supp['Staff Demand']

# Staff
ax = df_supp.plot.bar(y=['Staff Demand', 'Staff Supply'], figsize = (30, 10), fill=True, color=['black', 'red'],fontsize=16)
df_supp.plot(y=['Extra_Ressources'], color=['blue'], secondary_y = True, ax = ax, linewidth = 3,fontsize=16)
plt.title('Workforce: Demand vs. Supply',fontsize=16)
plt.xlabel('Day of the week',fontsize=16)
plt.ylabel('Number of Workers',fontsize=16)
plt.show()

st.pyplot(plt)


st.subheader('Schedule')
st.write(df_sch)

#st.subheader('Schedule')
#st.write(df_supp)
