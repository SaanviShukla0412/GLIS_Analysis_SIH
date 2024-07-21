from http.client import GONE
from turtle import goto
from flask import Flask, got_request_exception, redirect, render_template, request
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from joblib import load
from pyparsing import GoToColumn
import plotly.express as px
import plotly.graph_objects as go
import dash
import pickle


app = Flask(__name__)


dash_app = Dash(__name__, server=app, url_base_pathname='/population/')
hosp_app = Dash(__name__, server=app, url_base_pathname='/hospital/')
primary_app = Dash(__name__, server=app, url_base_pathname='/primaryhealth/')
crop_app = Dash(__name__, server=app, url_base_pathname='/cropyield/')
fact_app = Dash(__name__, server=app, url_base_pathname='/factorydata/')
hotel_app = Dash(__name__, server=app, url_base_pathname='/hoteldata/')
water_app = Dash(__name__, server=app, url_base_pathname='/waterdata/')


# Loading models
model = load('floods.save')
sc = load('transform.save')
aqi_model = pickle.load(open('model.pkl', 'rb'))
aqi_le = load('label_values')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/coal')
def coal():
    return render_template("coal_data.html")

@app.route('/inner-page')
def page():
    return render_template('inner-page.html')

@app.route('/moc')
def moc():
    return render_template('moc.html')

@app.route('/thermal')
def thermal():
    return render_template('coal_thermal.html')

@app.route("/aqi")
def aqi():
    return render_template("aqi_pred.html")

#Air Quality Prediction


@app.route("/aqi_predict", methods=['POST'])
def aqi_pred():
    if request.method == 'POST':
        city = request.form["city"]
        pm25 = request.form["pm25"]
        pm10 = request.form["pm10"]
        no = request.form["no"]
        no2 = request.form["no2"]
        nox = request.form["nox"]
        nh3 = request.form["nh3"]
        co = request.form["co"]
        so2 = request.form["so2"]
        o3 = request.form["o3"]
        benzene = request.form["benzene"]
        toluene = request.form["toluene"]
        xylene = request.form["xylene"]
        date = request.form["date"]

        city = aqi_le.transform([city])
        print(city[0])

        year = date.split('-')[0]
        month = date.split('-')[1]

        feature_cols = ['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                        'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'Year', 'Month']

        data = pd.DataFrame([[city[0], pm25, pm10, no, no2, nox, nh3, co, so2,
                            o3, benzene, toluene, xylene, year, month]], columns=feature_cols)
        print(data)
        pred_aqi = aqi_model.predict(data)

        if (pred_aqi[0] >= 0 and pred_aqi <= 50):
            res = 'GOOD'
        elif (pred_aqi[0] >= 50 and pred_aqi <= 100):
            res = 'SATISFACTORY'
        elif (pred_aqi[0] >= 100 and pred_aqi <= 200):
            res = 'MODERATELY POLLUTED'
        elif (pred_aqi[0] >= 200 and pred_aqi <= 300):
            res = 'POOR'
        elif (pred_aqi[0] >= 300 and pred_aqi <= 400):
            res = 'VERY POOR'
        else:
            res = 'SEVERE'

        return render_template('aqi_pred.html', prediction_aqi=res)
    

@app.route("/flood")
def flood():
    return render_template("flood_pred.html")


@app.route("/flood_predict", methods=['POST'])
def flood_pred():
    temp = request.form['temp']
    Hum = request.form['Hum']
    db = request.form['db']
    ap = request.form['ap']
    aa1 = request.form['aa1']

    data = [[float(temp), float(Hum), float(db), float(ap), float(aa1)]]
    prediction = model.predict(sc.transform(data))
    output = prediction[0]
    if (output == 0):
        return render_template('flood_pred.html', prediction='No possibility of severe flood')
    else:
        return render_template('flood_pred.html', prediction='possibility of severe flood')



# Population_dashboard


df = pd.read_csv('dataset\cities_r2.csv')

# Convert relevant columns to numeric types
numeric_columns = ['population_total', 'population_male', 'population_female', '0-6_population_total', '0-6_population_male', '0-6_population_female',
                   'literates_total', 'literates_male', 'literates_female', 'sex_ratio', 'child_sex_ratio',
                   'effective_literacy_rate_total', 'effective_literacy_rate_male', 'effective_literacy_rate_female',
                   'total_graduates', 'male_graduates', 'female_graduates']

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Handle missing values by dropping rows with missing values
df.dropna(subset=numeric_columns, inplace=True)

# Define the layout of the dashboard with styling enhancements
dash_app.layout = html.Div(
    style={
        'font-family': 'Arial, sans-serif',
        'margin': '20px',
        'padding': '20px',
        'background-color': '#f5f5f5',
    },
    children=[
        html.H1(
            "Population and Education Distribution Dashboard",
            style={
                'text-align': 'center',
                'color': '#333333',
                'font-size': '32px',
            },
        ),
        dcc.Dropdown(
            id='dropdown-city',
            options=[
                {'label': city, 'value': city} for city in df['name_of_city'].unique()
            ],
            value=df['name_of_city'].unique()[0],
            style={
                'width': '50%',
                'display': 'inline-block',
                'margin-bottom': '20px',
                'background-color': '#f0f0f0',
            },
        ),
        html.Div(
            children=[
                dcc.Graph(
                    id='bar-chart', style={'width': '33%', 'display': 'inline-block'}
                ),
                dcc.Graph(
                    id='bar-male-female',
                    style={'width': '33%', 'display': 'inline-block'},
                ),
                dcc.Graph(
                    id='bar-literate',
                    style={'width': '33%', 'display': 'inline-block'},
                ),
                dcc.Graph(
                    id='pie-chart-sex-ratio',
                    style={'width': '50%', 'display': 'inline-block'},
                ),
                dcc.Graph(
                    id='pie-chart-male-age-group',
                    style={'width': '50%', 'display': 'inline-block'},
                ),
                dcc.Graph(
                    id='pie-chart-female-age-group',
                    style={'width': '50%', 'display': 'inline-block'},
                ),
                dcc.Graph(
                    id='pie-chart-graduate-gender',
                    style={'width': '50%', 'display': 'inline-block'},
                ),
                dcc.Graph(
                    id='pie-chart-literate-gender',
                    style={'width': '50%', 'display': 'inline-block'},
                ),
            ]
        ),
    ],
)

# Define callbacks to update charts based on user input


@dash_app.callback(
    Output('bar-chart', 'figure'),
    Output('bar-male-female', 'figure'),
    Output('bar-literate', 'figure'),
    Output('pie-chart-sex-ratio', 'figure'),
    Output('pie-chart-male-age-group', 'figure'),
    Output('pie-chart-female-age-group', 'figure'),
    Output('pie-chart-graduate-gender', 'figure'),
    Output('pie-chart-literate-gender', 'figure'),
    Input('dropdown-city', 'value'),
)
def update_charts(selected_city):
    # Bar Chart for Graduate Distribution
    bar_chart = px.bar(
        df[df['name_of_city'] == selected_city],
        x='location',
        y=['total_graduates', 'male_graduates', 'female_graduates'],
        title=f'Graduate Distribution in {selected_city}',
        barmode='group',
        labels={'value': 'Count'},
        template='plotly_dark',
        color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71'],
    )

    # Bar Chart for Total Number of Male and Female
    bar_male_female = px.bar(
        df[df['name_of_city'] == selected_city],
        x='location',
        y=['population_male', 'population_female'],
        title=f'Total Number of Male and Female in {selected_city}',
        barmode='group',
        labels={'value': 'Count'},
        template='plotly_dark',
        color_discrete_sequence=['#3498db', '#e74c3c'],
    )

    # Bar Chart for Literate Males and Females
    bar_literate = px.bar(
        df[df['name_of_city'] == selected_city],
        x='location',
        y=['literates_male', 'literates_female'],
        title=f'Literate Males and Females in {selected_city}',
        barmode='group',
        labels={'value': 'Count'},
        template='plotly_dark',
        color_discrete_sequence=['#3498db', '#e74c3c'],
    )

    # Pie Chart for Sex Ratio
    pie_chart_sex_ratio = px.pie(
        df[df['name_of_city'] == selected_city],
        names=['Male', 'Female'],
        values=[df.loc[df['name_of_city'] == selected_city, 'population_male'].sum(),
                df.loc[df['name_of_city'] == selected_city, 'population_female'].sum()],
        title=f'Sex Ratio in {selected_city}',
        template='plotly_dark',
        color_discrete_sequence=['#3498db', '#e74c3c'],
    )

    # Pie Chart for Male Population Age Group Distribution
    pie_chart_male_age_group = px.pie(
        df[df['name_of_city'] == selected_city],
        names=['0-6', '7+'],
        values=[
            df.loc[df['name_of_city'] == selected_city,
                   '0-6_population_male'].sum(),
            df.loc[df['name_of_city'] == selected_city, 'population_male'].sum()
            - df.loc[df['name_of_city'] == selected_city,
                     '0-6_population_male'].sum(),
        ],
        title=f'Male Population Age Group Distribution in {selected_city}',
        template='plotly_dark',
        color_discrete_sequence=['#3498db', '#e74c3c'],
    )

    # Pie Chart for Female Population Age Group Distribution
    pie_chart_female_age_group = px.pie(
        df[df['name_of_city'] == selected_city],
        names=['0-6', '7+'],
        values=[
            df.loc[df['name_of_city'] == selected_city,
                   '0-6_population_female'].sum(),
            df.loc[df['name_of_city'] == selected_city,
                   'population_female'].sum()
            - df.loc[df['name_of_city'] == selected_city,
                     '0-6_population_female'].sum(),
        ],
        title=f'Female Population Age Group Distribution in {selected_city}',
        template='plotly_dark',
        color_discrete_sequence=['#3498db', '#e74c3c'],
    )

    # Pie Chart for Graduate Gender Distribution
    pie_chart_graduate_gender = px.pie(
        df[df['name_of_city'] == selected_city],
        names=['Male Graduates', 'Female Graduates'],
        values=[
            df.loc[df['name_of_city'] == selected_city, 'male_graduates'].sum(),
            df.loc[df['name_of_city'] == selected_city,
                   'female_graduates'].sum(),
        ],
        title=f'Graduate Gender Distribution in {selected_city}',
        template='plotly_dark',
        color_discrete_sequence=['#3498db', '#e74c3c'],
    )

    # Pie Chart for Literate Gender Distribution
    pie_chart_literate_gender = px.pie(
        df[df['name_of_city'] == selected_city],
        names=['Male Literates', 'Female Literates'],
        values=[
            df.loc[df['name_of_city'] == selected_city, 'literates_male'].sum(),
            df.loc[df['name_of_city'] == selected_city,
                   'literates_female'].sum(),
        ],
        title=f'Literate Gender Distribution in {selected_city}',
        template='plotly_dark',
        color_discrete_sequence=['#3498db', '#e74c3c'],
    )

    return (

        pie_chart_sex_ratio,
        pie_chart_male_age_group,
        pie_chart_female_age_group,
        pie_chart_graduate_gender,
        pie_chart_literate_gender,
        bar_chart, bar_male_female,
        bar_literate,
    )
    

# Hospital_Availability


# Load your population dataset
# Replace 'your_dataset.csv' with the path to your CSV file
hosp = pd.read_csv('dataset\govthospitalbeds2013jan.csv')

# Install necessary libraries
# Install necessary libraries
# pip install dash pandas plotly


# Install necessary libraries
# pip install dash pandas plotly

hosp.columns = hosp.columns.str.strip()

# Convert relevant columns to numeric types
numeric_columns = hosp.select_dtypes(include='number').columns
hosp[numeric_columns] = hosp[numeric_columns].apply(
    pd.to_numeric, errors='coerce')

# Handle missing values by dropping rows with missing values
hosp.dropna(subset=numeric_columns, inplace=True)

# Set up color palette
colors1 = {
    'background': '#f7f7f7',
    'text': '#333333',
    'header': '#2196F3',
    'chart_bg': '#ffffff',
    'chart_border': '#dddddd',
    # Add more colors as needed
    'param_colors': ['#3498db', '#e74c3c', '#2ecc71'],
}


# Define the layout of the dashboard with styling enhancements
hosp_app.layout = html.Div(
    style={
        'font-family': 'Arial, sans-serif',
        'background-color': colors1['background'],
        'padding': '20px',
        'border-radius': '10px',
        'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
        'max-width': '900px',
        'margin': 'auto',
    },
    children=[
        html.H1(
            "Hospital and Bed Availability  Dashboard",
            style={
                'text-align': 'center',
                'color': colors1['header'],
                'font-size': '36px',
                'margin-bottom': '20px',
                'text-shadow': '2px 2px 4px rgba(0, 0, 0, 0.2)',
            },
        ),
        dcc.Dropdown(
            id='dropdown-state',
            options=[{'label': state, 'value': state}
                     for state in hosp['State/UT/Division'].unique()],
            value=hosp['State/UT/Division'].unique()[0],
            style={
                'width': '100%',
                'margin-bottom': '20px',
                'background-color': colors1['chart_bg'],
                'border': '1px solid ' + colors1['chart_border'],
                'border-radius': '5px',
            },
        ),
        html.Div(
            children=[
                dcc.Graph(id='bar-hospitals',
                          config={'displayModeBar': False}),
                dcc.Graph(id='bar-beds', config={'displayModeBar': False}),
                dcc.Graph(id='bar-population',
                          config={'displayModeBar': False}),
                dcc.Graph(id='hospital-type-pie',
                          config={'displayModeBar': False}),
                dcc.Graph(id='bed-type-pie', config={'displayModeBar': False}),
            ],
            style={'display': 'flex', 'flex-wrap': 'wrap',
                   'justify-content': 'space-between'},
        ),
    ],
)

# Define callbacks to update charts based on user input


@hosp_app.callback(
    Output('bar-hospitals', 'figure'),
    Output('bar-beds', 'figure'),
    Output('bar-population', 'figure'),
    Output('hospital-type-pie', 'figure'),
    Output('bed-type-pie', 'figure'),
    Input('dropdown-state', 'value'),
)
def update_charts(selected_state):
    # Bar Chart for Number of Hospitals
    hospitals_columns = [
        'Number of Rural Hospitals (Govt.)', 'Number of Urban Hospitals (Govt.)', 'Number of Total Hospitals (Govt.)']
    bar_hospitals = px.bar(
        hosp[hosp['State/UT/Division'] == selected_state],
        x='Reference Period',
        y=hospitals_columns,
        title=f'Number of Hospitals in {selected_state}',
        barmode='group',
        labels={'value': 'Count'},
        template='plotly_dark',
        color_discrete_sequence=colors1['param_colors'],
    )

    # Bar Chart for Number of Beds
    beds_columns = [
        'Number of beds in Rural Hospitals (Govt.)', 'Number of beds in Urban Hospitals (Govt.)', 'Number of beds in Total Hospitals (Govt.)']
    bar_beds = px.bar(
        hosp[hosp['State/UT/Division'] == selected_state],
        x='Reference Period',
        y=beds_columns,
        title=f'Number of Beds in Hospitals in {selected_state}',
        barmode='group',
        labels={'value': 'Count'},
        template='plotly_dark',
        color_discrete_sequence=colors1['param_colors'],
    )

    # Bar Chart for Average Population Served
    avg_population_columns = ['Average Population Served Per Govt. Hospital',
                              'Average Population Served Per Govt. Hospital Bed']
    bar_population = px.bar(
        hosp[hosp['State/UT/Division'] == selected_state],
        x='Reference Period',
        y=avg_population_columns,
        title=f'Average Population Served in {selected_state}',
        barmode='group',
        labels={'value': 'Count'},
        template='plotly_dark',
        color_discrete_sequence=colors1['param_colors'],
    )

    # Pie chart to show the percentage of hospital types
    pie_data_hospital = hosp[hosp['State/UT/Division'] == selected_state]
    labels_hospital = ['Rural Hospitals', 'Urban Hospitals']
    values_hospital = [pie_data_hospital['Number of Rural Hospitals (Govt.)'].sum(
    ), pie_data_hospital['Number of Urban Hospitals (Govt.)'].sum()]
    fig_pie_hospital = go.Figure(
        data=[go.Pie(labels=labels_hospital, values=values_hospital, hole=0.3)])
    fig_pie_hospital.update_layout(
        title_text=f'Hospital Types in {selected_state}')

    # Pie chart to show the percentage of bed types
    pie_data_beds = hosp[hosp['State/UT/Division'] == selected_state]
    labels_beds = ['Beds in Rural Hospitals', 'Beds in Urban Hospitals']
    values_beds = [pie_data_beds['Number of beds in Rural Hospitals (Govt.)'].sum(
    ), pie_data_beds['Number of beds in Urban Hospitals (Govt.)'].sum()]
    fig_pie_beds = go.Figure(
        data=[go.Pie(labels=labels_beds, values=values_beds, hole=0.3)])
    fig_pie_beds.update_layout(
        title_text=f'Number of Beds in Urban and Rural Hospitals in {selected_state}')

    return fig_pie_hospital, bar_hospitals, fig_pie_beds, bar_beds, bar_population,

# Pimary heath -->


# Load your health facilities dataset
# Replace 'your_dataset.csv' with the path to your CSV file
primary = pd.read_csv('dataset\primaryheath.csv')

# Install necessary libraries
# pip install dash pandas plotly

# Install necessary libraries
# pip install dash pandas plotly


# Install necessary libraries
# pip install dash pandas plotly


# Set up color palette
colors = {
    'background': '#f7f7f7',
    'text': '#333333',
    'header': '#2196F3',
    'chart_bg': '#ffffff',
    'chart_border': '#dddddd',
}

# Define the layout of the dashboard with styling enhancements
primary_app.layout = html.Div(
    style={
        'font-family': 'Arial, sans-serif',
        'background-color': colors['background'],
        'padding': '20px',
        'border-radius': '10px',
        'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
        'max-width': '900px',
        'margin': 'auto',
    },
    children=[
        html.H1(
            "Primary Health Centers Dashboard",
            style={
                'text-align': 'center',
                'color': colors['header'],
                'font-size': '36px',
                'margin-bottom': '20px',
                'text-shadow': '2px 2px 4px rgba(0, 0, 0, 0.2)',
            },
        ),
        dcc.Dropdown(
            id='dropdown-state',
            options=[{'label': state, 'value': state}
                     for state in primary['State/UT'].unique()],
            value=primary['State/UT'].unique()[0],
            style={
                'width': '100%',
                'margin-bottom': '20px',
                'background-color': colors['chart_bg'],
                'border': '1px solid ' + colors['chart_border'],
                'border-radius': '5px',
            },
        ),
        dcc.Graph(id='bar-chart', config={'displayModeBar': False}),
        dcc.Graph(id='bubble-plot', config={'displayModeBar': False}),
    ],
)

# Define callback to update bar chart based on user input


@primary_app.callback(
    Output('bar-chart', 'figure'),
    Input('dropdown-state', 'value'),
)
def update_bar_chart(selected_state):
    selected_data = primary[primary['State/UT'] == selected_state]

    bar_chart = px.bar(
        selected_data,
        x=['Number of PHCs functioning with 4+ doctors',
           'Number of PHCs functioning with 3 doctors',
           'Number of PHCs functioning with 2 doctors',
           'Number of PHCs functioning with 1 doctor',
           'Number of PHCs functioning without doctor',
           'Number of PHCs functioning without lab tech',
           'Number of PHCs functioning without pharma',
           'Number of PHCs functioning with lady doctor'],
        y='Total PHCs functioning',  # Use 'Total PHCs' on the y-axis
        title=f'PHC Functioning Breakdown in {selected_state}',
        labels={'Total PHCs': 'Total PHCs'},
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Set3,  # Use qualitative color palette
    )

    return bar_chart

# Define callback to update bubble plot based on user input


@primary_app.callback(
    Output('bubble-plot', 'figure'),
    Input('dropdown-state', 'value'),
)
def update_bubble_plot(selected_state):
    selected_data = primary[primary['State/UT'] == selected_state]

    bubble_plot = px.scatter(
        selected_data,
        x='Number of PHCs functioning with 4+ doctors',
        y='Total PHCs functioning',
        size='Number of PHCs functioning with 2 doctors',  # Bubble size
        color='Number of PHCs functioning with 3 doctors',  # Bubble color
        title=f'Bubble Plot: PHC Functioning in {selected_state}',
        labels={'Number of PHCs functioning with 4+ doctors': 'PHCs with 4+ Doctors',
                'Total PHCs': 'Total PHCs',
                'Number of PHCs functioning with 2 doctors': 'PHCs with 2 Doctors',
                'Number of PHCs functioning with 3 doctors': 'PHCs with 3 Doctors'},
        template='plotly_dark',
        color_continuous_scale='Viridis',  # You can change the color scale
    )

    return bubble_plot

# Crop data


# Read the CSV file into a DataFrame
  # Replace with the actual path
crop = pd.read_csv('dataset\crop_yield.csv')

# Conversion factor for yield from quintals per hectare to kilograms per hectare
conversion_factor = 1000

# Convert the 'Yield' column to kilograms per hectare
crop['Yield_kg_per_hectare'] = crop['Yield'] * conversion_factor

# Define the layout of the dashboard with styling
crop_app.layout = html.Div(style={'backgroundColor': 'black', 'color': 'white'}, children=[
    html.H1(children='Crop Data Dashboard', style={
            'textAlign': 'center', 'color': '#0074D9'}),

    # Dropdown menus to select the Crop and State for the line chart
    dcc.Dropdown(
        id='crop-dropdown',
        options=[{'label': crop, 'value': crop}
                 for crop in crop['Crop'].unique()],
        # Set default value to the first crop in the dataset
        value=crop['Crop'].iloc[0],
        multi=False,
        style={'width': '50%', 'margin': '10px auto',
               'color': 'black'}  # Dropdown text color
    ),
    dcc.Dropdown(
        id='state-dropdown',
        options=[{'label': state, 'value': state}
                 for state in crop['State'].unique()],
        # Set default value to the first state in the dataset
        value=crop['State'].iloc[0],
        multi=False,
        style={'width': '50%', 'margin': '10px auto',
               'color': 'black'}  # Dropdown text color
    ),

    # Line chart based on the selected Crop and State
    dcc.Graph(
        id='line-chart',
        style={'width': '80%', 'margin': '20px auto'}
    ),

    # Scatter plot to show correlation between 'Area' and 'Yield_kg_per_hectare' aggregated by year
    dcc.Graph(
        id='scatter-plot',
        style={'width': '80%', 'margin': '20px auto'}
    ),

    # Bar chart based on the selected Crop and State
    dcc.Graph(
        id='bar-chart',
        style={'width': '80%', 'margin': '20px auto'}
    ),

    # Bubble chart to show the correlation between 'Area' and 'Yield_kg_per_hectare'
    dcc.Graph(
        id='bubble-chart',
        style={'width': '80%', 'margin': '20px auto'}
    )
])

# Define the callback to update the line chart, scatter plot, bar chart, and bubble chart based on the selected Crop and State


@crop_app.callback(
    [Output('line-chart', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('bar-chart', 'figure'),
     Output('bubble-chart', 'figure')],
    [Input('crop-dropdown', 'value'),
     Input('state-dropdown', 'value')]
)
def update_charts(selected_crop, selected_state):
    filtered_df = crop[(crop['Crop'] == selected_crop) &
                       (crop['State'] == selected_state)]

    # Line chart
    line_fig = px.line(filtered_df, x='Crop_Year', y='Yield_kg_per_hectare', line_group='State',
                       color='State', markers=True, title=f'{selected_crop} Production in {selected_state}',
                       labels={'Yield_kg_per_hectare': 'Production (kg/ha)'}, color_discrete_sequence=px.colors.qualitative.Set1)
    line_fig.update_layout(plot_bgcolor='black',
                           paper_bgcolor='black', font_color='white')

    # Scatter plot aggregated by year
    scatter_fig = px.scatter(filtered_df, x='Crop_Year', y='Yield_kg_per_hectare', color='State',
                             title=f'Correlation: Yield vs. Year for {selected_crop} in {selected_state}',
                             labels={
                                 'Yield_kg_per_hectare': 'Production (kg/ha)', 'Crop_Year': 'Year'},
                             color_discrete_sequence=px.colors.qualitative.Set1)
    scatter_fig.update_layout(plot_bgcolor='black',
                              paper_bgcolor='black', font_color='white')

    # Bar chart for total production by state and crop
    bar_fig = px.bar(filtered_df, x='Crop_Year', y='Yield_kg_per_hectare', color='State',
                     title=f'Total Production by Crop and State ({selected_crop} in {selected_state})',
                     labels={
                         'Yield_kg_per_hectare': 'Total Production (kg/ha)', 'Crop_Year': 'Year'},
                     color_discrete_sequence=px.colors.qualitative.Set1)
    bar_fig.update_layout(plot_bgcolor='black',
                          paper_bgcolor='black', font_color='white')

    # Bubble chart for correlation between 'Area' and 'Yield_kg_per_hectare'
    bubble_fig = px.scatter(filtered_df, x='Area', y='Yield_kg_per_hectare', size='Production', color='Crop_Year',
                            title=f'Correlation: Area vs. Yield (Bubble Chart) for {selected_crop} in {selected_state}',
                            labels={
                                'Yield_kg_per_hectare': 'Production (kg/ha)', 'Area': 'Area (ha)', 'Crop_Year': 'Year'},
                            color_discrete_sequence=px.colors.qualitative.Set1)
    bubble_fig.update_layout(plot_bgcolor='black',
                             paper_bgcolor='black', font_color='white')

    return line_fig, scatter_fig, bar_fig, bubble_fig

# Factory Data


fact = pd.read_csv('dataset\FACTORY_EMPLOYMENT.csv')

fact_app.layout = html.Div(style={'backgroundColor': 'black', 'color': 'white', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.Img(src='your_logo.png', style={
             'width': '10%', 'margin': '10px auto'}),  # Add your logo

    html.H1(children='Factories Data Dashboard', style={
            'textAlign': 'center', 'color': '#1F77B4', 'fontSize': 36, 'marginBottom': 20}),

    # Dropdown menus to select the X and Y axes for the bar chart
    dcc.Dropdown(
        id='x-axis-dropdown',
        options=[{'label': col, 'value': col} for col in fact.columns],
        value='Year',
        multi=False,
        style={'width': '60%', 'margin': '10px auto', 'fontSize': 18, }
    ),
    dcc.Dropdown(
        id='y-axis-dropdown',
        options=[{'label': col, 'value': col} for col in fact.columns],
        value='Total (Private Sector)',
        multi=False,
        style={'width': '60%', 'margin': '10px auto', 'fontSize': 18}
    ),

    # Bar chart based on the selected X and Y axes
    dcc.Graph(
        id='bar-chart',
        style={'width': '80%', 'margin': '20px auto',
               'backgroundColor': '#333333'}
    ),

    # Line chart based on the selected X and Y axes
    dcc.Graph(
        id='line-chart',
        style={'width': '80%', 'margin': '20px auto',
               'backgroundColor': '#333333'}
    ),

    # Scatter plot based on the selected X and Y axes
    dcc.Graph(
        id='scatter-plot',
        style={'width': '80%', 'margin': '20px auto',
               'backgroundColor': '#333333'}
    )
])

# Define the callbacks to update the charts based on the selected X and Y axes


@fact_app.callback(
    Output('bar-chart', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_bar_chart(x_axis, y_axis):
    fig = px.bar(fact, x=x_axis, y=y_axis, title='Factories Data Bar Chart')
    fig.update_layout(
        plot_bgcolor='#333333',
        paper_bgcolor='#333333',
        font_color='white',
        xaxis=dict(gridcolor='#555555'),
        yaxis=dict(gridcolor='#555555'),
    )
    return fig


@fact_app.callback(
    Output('line-chart', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_line_chart(x_axis, y_axis):
    fig = px.line(fact, x=x_axis, y=y_axis, title='Factories Data Line Chart')
    fig.update_layout(
        plot_bgcolor='#333333',
        paper_bgcolor='#333333',
        font_color='white',
        xaxis=dict(gridcolor='#555555'),
        yaxis=dict(gridcolor='#555555'),
    )
    return fig


@fact_app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_scatter_plot(x_axis, y_axis):
    fig = px.scatter(fact, x=x_axis, y=y_axis,
                     title='Factories Data Scatter Plot')
    fig.update_layout(
        plot_bgcolor='#333333',
        paper_bgcolor='#333333',
        font_color='white',
        xaxis=dict(gridcolor='#555555'),
        yaxis=dict(gridcolor='#555555'),
    )
    return fig

# Hotel Data


hotel = pd.read_csv('dataset\HotelTourismData.csv')

# Filter out null values in the 'STATE' column
hotel = hotel.dropna(subset=['STATE'])

# Define the layout of the dashboard with styling
hotel_app.layout = html.Div(style={'backgroundColor': 'black', 'color': 'white', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.Img(src='your_logo.png', style={
             'width': '10%', 'margin': '10px auto'}),  # Add your logo

    html.H1(children='Hotel Data Dashboard', style={
            'textAlign': 'center', 'color': '#1F77B4', 'fontSize': 36, 'marginBottom': 20}),

    # Option section to select the state
    html.Div([
        html.Label('Select State:', style={'fontSize': 18}),
        dcc.Dropdown(
            id='state-dropdown',
            options=[{'label': state, 'value': state}
                     for state in hotel['STATE'].unique()],
            # Set default value to the first state
            value=hotel['STATE'].unique()[0],
            multi=False,
            style={'width': '60%', 'marginBottom': 20, 'fontSize': 16,
                   'color': '#2A3F54', 'backgroundColor': '#FFFFFF'}
        ),
    ]),

    # Bar chart displaying the number of rooms for each hotel in the selected state
    dcc.Graph(
        id='bar-chart',
        style={'width': '80%', 'margin': '20px auto',
               'backgroundColor': 'black'}
    ),

    # Pie chart displaying the distribution of rooms in the selected state
    dcc.Graph(
        id='pie-chart',
        style={'width': '60%', 'margin': '20px auto',
               'backgroundColor': 'black'}
    )
])

# Define the callback to update the bar chart based on the selected state


@hotel_app.callback(
    Output('bar-chart', 'figure'),
    [Input('state-dropdown', 'value')]
)
def update_bar_chart(selected_state):
    filtered_df = hotel[hotel['STATE'] == selected_state]
    fig = px.bar(filtered_df, x='HOTEL NAME', y='Rooms',
                 title=f'Number of Rooms in {selected_state} Hotels')
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        xaxis=dict(title='Hotel Name', tickangle=-45, tickfont=dict(size=12)),
        yaxis=dict(title='Number of Rooms'),
    )
    # Change the bar color to a different color (e.g., teal)
    fig.update_traces(marker_color='#008080')
    return fig

# Define the callback to update the pie chart based on the selected state


@hotel_app.callback(
    Output('pie-chart', 'figure'),
    [Input('state-dropdown', 'value')]
)
def update_pie_chart(selected_state):
    filtered_df = hotel[hotel['STATE'] == selected_state]
    fig = px.pie(filtered_df, names='HOTEL NAME', values='Rooms',
                 title=f'Distribution of Rooms in {selected_state} Hotels')
    fig.update_traces(marker=dict(
        colors=['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']))
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
    )
    return fig


# Water Level

water = pd.read_csv('dataset\waterlevel.csv')

# Filter out null values
water = water.dropna(subset=['District', 'Year', 'Water Level'])

# Unique options for dropdowns
district_options = [{'label': District, 'value': District}
                    for District in water['District'].unique()]
year_options = [{'label': year, 'value': year}
                for year in water['Year'].unique()]

# Define the layout of the dashboard with styling
water_app.layout = html.Div(style={'backgroundColor': 'white', 'color': 'black', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.H1(children='Water Level Dashboard', style={
            'textAlign': 'center', 'color': '#1F77B4', 'fontSize': 36, 'marginBottom': 20}),

    # Option section to select the district
    html.Div([
        html.Label('Select District:', style={'fontSize': 18}),
        dcc.Dropdown(
            id='district-dropdown',
            options=district_options,
            # Set default value to the first district
            value=district_options[0]['value'],
            multi=False,
            style={'width': '60%', 'marginBottom': 20, 'fontSize': 16}
        ),
    ]),

    # Line chart displaying water levels over the years for the selected district
    dcc.Graph(
        id='line-chart',
        style={'width': '80%', 'margin': '20px auto'}
    ),

    # Option section to select the year
    html.Div([
        html.Label('Select Year:', style={'fontSize': 18}),
        dcc.Dropdown(
            id='year-dropdown',
            options=year_options,
            # Set default value to the first year
            value=year_options[0]['value'],
            multi=False,
            style={'width': '60%', 'marginBottom': 20, 'fontSize': 16}
        ),
    ]),

    # Bar chart displaying the actual water level for each district across all years
    dcc.Graph(
        id='bar-chart',
        style={'width': '80%', 'margin': '20px auto'}
    )
])

# Define the callback to update the line chart based on the selected district


@water_app.callback(
    Output('line-chart', 'figure'),
    [Input('district-dropdown', 'value')]
)
def update_line_chart(selected_district):
    filtered_df = water[water['District'] == selected_district]
    fig = px.line(filtered_df, x='Year', y='Water Level',
                  title=f'Water Levels Over the Years in {selected_district}')
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Water Level'),
    )
    return fig

# Define the callback to update the bar chart based on the selected district


@water_app.callback(
    dash.dependencies.Output('bar-chart', 'figure'),
    [dash.dependencies.Input('district-dropdown', 'value')]
)
def update_bar_chart(selected_district):
    filtered_df = water[water['District'] == selected_district]
    fig = px.bar(
        filtered_df,
        x='Year',
        y='Water Level',
        title=f'Water Level Over the Years in {selected_district} District',
        labels={'Water Level': 'Water Level'},
        color='Year',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        hover_name='Year',
        hover_data={'Water Level'},
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Water Level'),
    )
    return fig



if __name__ == '__main__':
    app.run(debug=True , threaded=True)
