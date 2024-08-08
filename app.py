import pandas as pd
import streamlit as st
from PIL import Image
from model import preprocessing, load_model_and_make_scoring


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/crab.jpg')

    st.set_page_config(
        layout="centered",
        initial_sidebar_state="auto",
        page_title="Credit Scoring",
        page_icon=image,
    )

    st.write(
        """
        # Здесь можно получить прогноз о том, с какой вероятностью Вам 
        # одобрят кредит в абстрактном банке.
        """
    )

    st.image(image)


def output_user_data(df):
    st.write('## Ваши данные')
    st.write(df)


def output_prediction(prediction, prediction_probas):
    st.write('## Прогноз')
    st.write(prediction)

    st.write('## Веротяность получить от банка такой ответ')
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Ваши параметры')
    user_input_df = sidebar_input_features()

    df = pd.read_csv('data/credit_scoring.csv')
    new_df = pd.concat((user_input_df, df))
    X_train, X_test, y_train, y_test = preprocessing(new_df)
    
    user_X_train = X_train[:1]

    pred, pred_probas = load_model_and_make_scoring(user_X_train)
    output_prediction(pred, pred_probas)


def sidebar_input_features():
    SeriousDlqin2yrs = 1
    RevolvingUtilizationOfUnsecuredLines = st.sidebar.number_input('Общий баланс средств на Вашей карте, поделенный на желаемую сумму кретидного лимита', min_value=0.0, max_value=1.0, value=0.5)
    age = st.sidebar.slider('Возраст', min_value=0.0, max_value=100.0, value=25.0, step=1.0)
    NumberOfTime30_59DaysPastDueNotWorse = st.sidebar.number_input('Сколько раз за последние 2 года у Вас наблюдалась просрочка 30-59 дней?', min_value=0, max_value=24)
    DebtRatio = st.sidebar.number_input('Ежемесячные расходы в долларах, делённые на ежемесячный доход', min_value=0.0, max_value=1.0)
    MonthlyIncome = st.sidebar.number_input('Ежемесячный доход в долларах', min_value=0)
    NumberOfOpenCreditLinesAndLoans = st.sidebar.number_input('Количество открытых кредитов', min_value=0)
    NumberOfTimes90DaysLate= st.sidebar.number_input('Сколько раз за последние 2 года у Вас наблюдалась просрочка 90 и более дней?', min_value=0, max_value=9)
    NumberOfTime60_89DaysPastDueNotWorse = st.sidebar.number_input('Сколько раз за последние 2 года у Вас наблюдалась просрочка 60-89 дней?', min_value=0, max_value=12)
    NumberOfDependents = st.sidebar.slider('Число иждивенцев на попечении', min_value=0, max_value=100, value=0, step=1)
    RealEstateLoansOrLines = "a"
    GroupAge = "a"


    data = {
        "SeriousDlqin2yrs" : SeriousDlqin2yrs,
        "RevolvingUtilizationOfUnsecuredLines" : RevolvingUtilizationOfUnsecuredLines,
        "age" : age,
        "NumberOfTime30-59DaysPastDueNotWorse" : NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio" : DebtRatio,
        "MonthlyIncome" : MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans" : NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate" : NumberOfTimes90DaysLate,
        "NumberOfTime60-89DaysPastDueNotWorse" : NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents" : NumberOfDependents,
        "RealEstateLoansOrLines" : RealEstateLoansOrLines,
        "GroupAge" : GroupAge
    }

    df = pd.DataFrame(data, index=[0])

    print(df)

    return df


if __name__ == "__main__":
    process_main_page()



