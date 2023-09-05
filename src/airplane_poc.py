import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st
import datetime
import streamlit.components.v1 as components
import leafmap.foliumap as lf
from src.sms_alert import send_sms
from trackme import track
#import trackme

DEFAULT_MAP_SPECS = dict(height=450, width=600)  # in px
APP_TITLE = 'SURVEILLANCE'
LOGO = ""# "https://technatium.com/wp-content/uploads/2022/09/cropped-TECHNATIUM-LOGO-SANSFOND.png"


def add_code_logo(logowidth: str = "500px"):
    CODE_LOGO = "https://technatium.com/wp-content/uploads/2022/09/cropped-TECHNATIUM-LOGO-SANSFOND.png" #"https://www.pngitem.com/pimgs/m/77-779399_transparent-homework-icon-png-blue-code-icon-png.png"
    st.markdown(
        f'<img src="{CODE_LOGO}" width="{logowidth}"/>',
        unsafe_allow_html=True,
    )

#@st.cache(suppress_st_warning=True)
def add_title(uselogo: bool = True, logowidth: str='500px'):
    col1, col2, col3, col4= st.columns(4)
    with col1:
        st.write(f'# {APP_TITLE}')
    with col4:
        if uselogo:
            st.markdown(
                f'<img src="{LOGO}" width="{logowidth}"/>',
                unsafe_allow_html=True,
            )

def statistics(geojonfile):
    """
    :compute number of objects per category
    :plot histograms (multi bins) per date to show the evolution
    :return:
    """
    obj = pd.read_json(geojonfile)
    return obj


def plotgraphs(obj1t1,obj2t1,obj1t2,obj2t2,obj1t3,obj2t3):
    obj1t1 = pd.read_json(obj1t1)
    obj2t1 = pd.read_json(obj2t1)
    obj1t2 = pd.read_json(obj1t2)
    obj2t2 = pd.read_json(obj2t2 )
    obj1t3 = pd.read_json(obj1t3)
    obj2t3 = pd.read_json(obj2t3)
    df = pd.DataFrame(
    [["T1",obj1t1['type'].count(), obj2t1['type'].count()],
     ["T2", obj1t2['type'].count(), obj2t2['type'].count()],
     ["T3", obj1t3['type'].count(), obj2t3['type'].count()]],
    columns=[ "Time Series" , "Cars", "Planes"])
    #fig = px.bar(df, x="Time Series", y=["Cars", "Planes"], barmode='group', height=500)
    fig = px.bar(df, x="Time Series", y=["Cars", "Planes"], barmode='group', height=500,
                 color_discrete_map={'Cars': "#800000", 'Planes': "#0000ff"})
    # st.dataframe(df) # if need to display dataframe
    st.plotly_chart(fig)
    return obj1t1['type'].count(), obj2t1['type'].count(),  obj1t2['type'].count(), obj2t2['type'].count(), obj1t3['type'].count(), obj2t3['type'].count()


def plotgraph(obj1,obj2):
    obj1 = pd.read_json(obj1)
    obj2 = pd.read_json(obj2)
    df = pd.DataFrame(
    [["T1",obj1['type'].count(), obj2['type'].count()]],
    columns=[ "Detected Object" , "Cars", "Planes"])
    fig = px.bar(df, x="Detected Object", y=["Cars", "Planes"], barmode='group', height=500,
                 color_discrete_map={'Cars': "#800000", 'Planes': "#0000ff"})
    # st.dataframe(df) # if need to display dataframe "#800000", "#0000ff"
    st.plotly_chart(fig)
    return obj1['type'].count(), obj2['type'].count()

def alerts(content):
    """
    :return:
    """
    st.write('Alerts ZONE')
    #receiver_no = st.number_input('Insert a number')
    #st.write('The current number is ', receiver_no)
    alert = send_sms(['+33766121245'], content)
    #st.write(alert)
    #return None



def stat_bar(obj1t1, obj2t1, obj1t2, obj2t2, obj1t3, obj2t3):
    global v11, v21, v12, v22, v13, v23
    graph, alert = st.columns([6,3])
    with graph:
        st.success("Detection Status")
        #if option == "Luxemb Airport" or "Luxemb Airport - Static": # and not "Planes & Cars":
        v11, v21, v12, v22, v13, v23 = plotgraphs(obj1t1, obj2t1, obj1t2, obj2t2, obj1t3, obj2t3)
    with alert:
        st.success("Alert Status")
        #message = {"Number of Car at T3":v13, "Number of Plane at T3": v23}
        message = "{} Cars and {} Planes are Detected at T3.".format(v13,v23)
        #alerts(message)
        st.write(message)

def stat_bar2(cdgt1planes,cdgt1cars):
    graph, alert = st.columns([6,3])
    with graph:
        st.success("Detection Status")
        v, v1 = plotgraph(cdgt1planes,cdgt1cars)
    with alert:
        st.success("Alert Status")
        message = "{} Cars and {} Planes are Detected.".format(v,v1)
        #alerts(message)
        st.write(message)

def my_app(wide_layout:bool=False): #
    if wide_layout:
        #layout = 'wide'
        st.set_page_config(layout="wide", page_icon=LOGO, page_title=APP_TITLE)
        #st.set_page_config(page_icon=LOGO, page_title=APP_TITLE)
        add_title(uselogo=True, logowidth='250px')


    #    with a:
    #        d = st.date_input(
    #        "Choose a date",
    #        datetime.date(2023, 7, 6))
    #        st.write('You\'re viewing image:', d)



    map_specs = DEFAULT_MAP_SPECS


    #available_tiles = ["Planes & Cars", "Trees & Buildings", "Luxemb Airport", "Luxemb Airport - Static"]
    available_tiles = ["Airport I", "Airport I - Static", "Airport II",  "Tracker"]

    DEFAULT_TILES = "Luxemb Airport"
    tiles = dict()

    with st.sidebar:
        option = st.radio("Choose a site to monitor",
                          options= available_tiles)

    car_style = {
    "stroke": True,
    "color": "#800000",
    "weight": 2,
    "opacity": 1,
    "fill": True,
    "fillColor": "#0000ff",
    "fillOpacity": 0.1,}

    plane_style = {
    "stroke": True,
    "color": "#0000ff",
    "weight": 2,
    "opacity": 1,
    "fill": True,
    "fillColor": "#0000ff",
    "fillOpacity": 0.1,}

    trees_style = {
    "stroke": True,
    "color": "#008000",
    "weight": 2,
    "opacity": 1,
    "fill": True,
    "fillColor": "#0000ff",
    "fillOpacity": 0.1,}


    hover_style = {"fillOpacity": 0.7}
    cdgt1planes = "data/planes.geojson"
    cdgt1cars = "data/car.geojson"

    obj1t1 = "data/luxt1car.geojson"
    obj2t1 = "data/luxt1airplane.geojson"
    obj1t2 = "data/luxt2car.geojson"
    obj2t2 = "data/luxt2airplane.geojson"
    obj1t3 = "data/luxt3car.geojson"
    obj2t3 = "data/luxt3airplane.geojson"


    global v11, v21, v12, v22, v13, v23
    if option =="Airport I":
        #global selected_date
        List_of_date = ["T1", "T2", "T3"]
        a, b, c = st.columns([1,1,1])
        selected_date = a.selectbox('Select a surveillance date', List_of_date)
        m = lf.Map(center=[49.634341, 6.22118],zoom=15, google_map="SATELLITE") #SATELLITE , HYBRID
        if selected_date == "T1":
            m.add_geojson("data/luxt1car.geojson", layer_name="Cars", style=car_style, hover_style=hover_style)
            m.add_geojson("data/luxt1airplane.geojson", layer_name="Airplanes", style=plane_style, hover_style=hover_style)
        elif selected_date == "T2":
            m.add_geojson("data/luxt2car.geojson", layer_name="Cars", style=car_style, hover_style=hover_style)
            m.add_geojson("data/luxt2airplane.geojson", layer_name="Airplanes", style=plane_style, hover_style=hover_style)

        elif selected_date == "T3":
            m.add_geojson("data/luxt3car.geojson", layer_name="Cars", style=car_style, hover_style=hover_style)
            m.add_geojson("data/luxt3airplane.geojson", layer_name="Airplanes", style=plane_style, hover_style=hover_style)
        # Show map by loading html
        components.html(
            m.to_html(),
            width=map_specs['width'] * 2,
            height=map_specs['height'] * 1.5,
        )
        stat_bar(obj1t1, obj2t1, obj1t2, obj2t2, obj1t3, obj2t3)

    if option == "Airport I - Static":
        checkimg1, checkimg2, checkimg3 = st.columns([2,1,1])
        img1 = checkimg1.checkbox("Show T1", value=True)
        img2 = checkimg2.checkbox("Show T2")
        img3 = checkimg3.checkbox("Show T3")
        if img1:
            image = Image.open('data/LuxAP1.png')
            st.image(image, caption='T1')
        if img2:
            image = Image.open('data/LuxAP2.png')
            st.image(image, caption='T2')
        if img3:
            image = Image.open('data/LuxAP3.png')
            st.image(image, caption='T3')
        stat_bar(obj1t1, obj2t1, obj1t2, obj2t2, obj1t3, obj2t3)


    if option == "Airport II":
        m = lf.Map(center=[49.004448, 2.578354], zoom=16, google_map="SATELLITE")
        m.add_geojson(cdgt1planes, layer_name="Planes", style=plane_style, hover_style=hover_style)
        m.add_geojson(cdgt1cars, layer_name="Cars", style=car_style, hover_style=hover_style)
        # Show map by loading html
        components.html(
            m.to_html(),
            width=map_specs['width'] * 2,
            height=map_specs['height'] * 1.5,
        )
        stat_bar2(cdgt1planes,cdgt1cars)
    ################


    if option == "Tracker":
        track()
