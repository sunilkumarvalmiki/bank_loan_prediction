#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q pyngrok')

get_ipython().system('pip install -q streamlit')

get_ipython().system('pip install -q streamlit_ace')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', ' \nimport pickle\nimport streamlit as st\n \n# loading the trained model\npickle_in = open(\'classifier.pkl\', \'rb\') \nclassifier = pickle.load(pickle_in)\n \n@st.cache()\n  \n# defining the function which will make the prediction using the data which the user inputs \ndef prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   \n \n    # Pre-processing user input    \n    if Gender == "Male":\n        Gender = 0\n    else:\n        Gender = 1\n \n    if Married == "Unmarried":\n        Married = 0\n    else:\n        Married = 1\n \n    if Credit_History == "Unclear Debts":\n        Credit_History = 0\n    else:\n        Credit_History = 1  \n \n    LoanAmount = LoanAmount / 1000\n \n    # Making predictions \n    prediction = classifier.predict( \n        [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])\n     \n    if prediction == 0:\n        pred = \'Rejected\'\n    else:\n        pred = \'Approved\'\n    return pred\n      \n  \n# this is the main function in which we define our webpage  \ndef main():       \n    # front end elements of the web page \n    html_temp = """ \n    <div style ="background-color:yellow;padding:13px"> \n    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> \n    </div> \n    """\n      \n    # display the front end aspect\n    st.markdown(html_temp, unsafe_allow_html = True) \n      \n    # following lines create boxes in which user can enter data required to make prediction \n    Gender = st.selectbox(\'Gender\',("Male","Female"))\n    Married = st.selectbox(\'Marital Status\',("Unmarried","Married")) \n    ApplicantIncome = st.number_input("Applicants monthly income") \n    LoanAmount = st.number_input("Total loan amount")\n    Credit_History = st.selectbox(\'Credit_History\',("Unclear Debts","No Unclear Debts"))\n    result =""\n      \n    # when \'Predict\' is clicked, make the prediction and store it \n    if st.button("Predict"): \n        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) \n        st.success(\'Your loan is {}\'.format(result))\n        print(LoanAmount)\n     \nif __name__==\'__main__\': \n    main()')

