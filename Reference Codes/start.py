# %%
import panel as pn
pn.extension()
pn.extension('tabulator', 'plotly', 'notifications')
pn.extension(loading_spinner='dots', loading_color='#00aa41', sizing_mode="stretch_width")
pn.extension(notifications=True)
import itertools
import torch
from sentence_transformers import SentenceTransformer, util
from operator import itemgetter

import joblib
import pandas as pd
import numpy as np

from bokeh.io import show
from bokeh.models import ColumnDataSource

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

from html.parser import HTMLParser
from html.entities import name2codepoint
import datetime  as dt

from azure.identity import ManagedIdentityCredential, InteractiveBrowserCredential
from azure.keyvault.secrets import SecretClient

import snowflake.connector
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from sqlalchemy.dialects import registry

from bokeh.models.widgets.tables import TextEditor

# %%
credential = ManagedIdentityCredential()
#credential = InteractiveBrowserCredential(additionally_allowed_tenants = '*')
secret_client = SecretClient(vault_url="https://financeds.vault.azure.net/", credential=credential)

account = secret_client.get_secret("snowflake-account").value
user = secret_client.get_secret("snowflake-user").value
password = secret_client.get_secret("snowflake-password").value
database = secret_client.get_secret("snowflake-database").value
role = secret_client.get_secret("snowflake-role").value
warehouse = secret_client.get_secret("snowflake-warehouse").value

# %%
embeddings_2d = joblib.load('embeddings_2d.pkl')
df_embeddings, df_topics, df = joblib.load('synergi_tm_results.pkl')
df_topics = df_topics.dropna()

df_topics.loc[df_topics[0] == -1, 'label'] = 'unclear'

topics_unique = df_topics.groupby('label').head(1)[[0, 'label']]
df = pd.merge(df, topics_unique, how='inner', left_on='topic', right_on=0).drop(columns=['topic']).\
     rename(columns={'label': 'topic'})

df = df[['topic', 'CASENO',     'FUNCTIONAL_AREA',     'EMPLOYMENT_CATEGORY', 'CASE_TYPE',
       'CASE_SEVERITY',   'HAZARD_PHYSICAL_SECURITY_EVENT', 'FULL_INVESTIGATION_DONE',
       'CASE_OCCURENCE_DATE', 'PERSONAL_INJURIES', 'CASE_CLOSED_DATE',
       'LANGUAGE', 'LOCATION_SHORT', 'CASE_DESCRIPTION_EN']]

df['CASE_OCCURENCE_DATE'] = pd.to_datetime(df['CASE_OCCURENCE_DATE'])


topic_counts = df.groupby(['topic']).size().reset_index().rename(columns={'0': 'counts'})


model = SentenceTransformer('all-mpnet-base-v2')


embeddings = df_embeddings.values[: ,0:-1].astype(np.float32)
embeddings_casenos = df_embeddings.values[:, -1].astype(int)
embeddings_tensor = torch.tensor(embeddings)


filters_panel_table = {'FUNCTIONAL_AREA': {'type': 'list', 'valuesLookup': True},
                          'LOCATION_SHORT': {'type': 'list', 'valuesLookup': True},
                          'EMPLOYMENT_CATEGORY': {'type': 'list', 'valuesLookup': True},
                          'CASE_TYPE': {'type': 'list', 'valuesLookup': True},
                            'CASE_SEVERITY': {'type': 'list', 'valuesLookup': True},
                            'HAZARD_PHYSICAL_SECURITY_EVENT': {'type': 'list', 'valuesLookup': True},
                            'FULL_INVESTIGATION_DONE': {'type': 'list', 'valuesLookup': True},
                            'PERSONAL_INJURIES': {'type': 'list', 'valuesLookup': True}}

select_case_type = pn.widgets.MultiSelect(options = ['Safety','Process Safety', 'Environment','Operational loss','Asset and Reputation damage/loss','Information Security','Physical Security'], 
                                          name='Case Type Filter (CTRL + Click to multiselect)', size =7)



editors_panel_table = {}
for c in df.columns:
    editors_panel_table[c] = None

# %%
class HTMLParser(HTMLParser):
    
    def __init__(self):
        super().__init__()
        self.data_ = ''

    def handle_data(self, data):
        self.data_ +=  data

# %%
def search(event):
    search_results_table.selection = []
    if text_input.value == '<p><br></p>' or text_input.value == '' :
        search_results_table.header_filters = None
        search_results_table.value = return_empty_table()
        return
    
    parser = HTMLParser()
    parser.feed(text_input.value)   
    text = parser.data_
 
    
    tensor = model.encode(text)
    cos_scores = util.cos_sim(tensor, embeddings_tensor)[0]
    vals, indices = torch.topk(cos_scores, k=n_results_selector.value)

    case_nos = [embeddings_casenos[i] for i in indices]
    
    df_ = df.iloc[pd.Index(df['CASENO']).get_indexer(case_nos)].copy()
    df_['RANK'] = list(range(1, n_results_selector.value+1))
    
    cols = df_.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_ = df_[cols]
    
    search_results_table.value = df_
    search_results_table.header_filters = filters_panel_table
    search_results_table.add_filter(select_case_type, 'CASE_TYPE')

    
    
def return_text(row):    
    text_ = row['CASE_DESCRIPTION_EN']
    text = fit_text(text_, 130)
        
    n_lines = len(text.split('<br>')) + 1
    
    height = min(250, n_lines * 20)
    
    col = pn.Row(pn.Spacer(width=25) ,pn.Column(pn.pane.HTML(text, height=height, width=900), scroll=True, 
                                                 sizing_mode='fixed', 
                                                 width=940, height=height+25)
                )
    
    return col


def fit_text(text_, max_line_size):
    text_ = text_.replace('\n', '<br>')
    words = text_.split(' ')
    text = ''
    current_len = 0
    for i in range(len(words)):
        text += words[i] + ' '
        current_len += len(words[i]) + 1
        
        if '<br>' in words[i]:
            current_len = 0
        
        if current_len > max_line_size:
            current_len = 0
            text += '<br>'
            
    return text


def delete_query(event):
    text_input.value = ''
    
    
def return_empty_table():
    return pd.DataFrame(data=None, columns=df.columns, index=None)


def update_expanded_table(value):
    if len(search_results_table.expanded) > 3:
        search_results_table.expanded = search_results_table.expanded[1:]
        
    if len(topics_doc_table.expanded) > 3:
        topics_doc_table.expanded = topics_doc_table.expanded[1:]
        
    if len(recent_doc_table.expanded) > 1:
        recent_doc_table.expanded = recent_doc_table.expanded[1:]
        
    if len(recent_doc_neighbors_table.expanded) > 3:
        recent_doc_neighbors_table.expanded = recent_doc_neighbors_table.expanded[1:]

# %%
text_input = pn.widgets.TextEditor(placeholder='', height=300, width=500, toolbar=False, margin=5)
text_input.param.watch(search, 'value')

delete_query_button = pn.widgets.Button(name='delete query',  height=30, width=120,  margin=5, sizing_mode='fixed')
delete_query_button.on_click(delete_query)

n_results_selector = pn.widgets.Select(name='number results', options=list(range(10, 101, 10)), height=20, width=120, value=20,
                            sizing_mode='fixed', margin=5)
n_results_selector.param.watch(search, 'value')

text_search = pn.widgets.StaticText(name='Search in Synergi free texts', 
                                    value='''This a semantic search interface on the Synergi free text input. 
                                    It uses not only keywords within the search query but also tries to determine 
                                    the intent and contextual meaning behind a search query. <br />
                                    Try -for example- to search for "body arm leg". The search results will also 
                                    show entries with neck injuries and other non-overlapping texts.''')



search_results_table = pn.widgets.Tabulator(return_empty_table(), theme='simple', sizing_mode='stretch_width', 
                                            show_index=False, 
                                            editors=editors_panel_table, header_filters=None,
                                            selectable=True, layout='fit_data_stretch', width=None, 
                                            height=500, embed_content=False, row_content=return_text)

search_results_table.param.watch(update_expanded_table, 'expanded')


# %%
def submit_feedback_1(event):
    with pn.param.set_values(feedback_search, loading=True):
        engine = create_engine(URL(
                account = account,
                user = user,
                password = password,
                database = database,
                schema = 'sinergi',
                role = role,
                warehouse = warehouse
            ))
            
        relevant = feedback_checkbox_1.value
        comment = comment_textinput_1.value
        query = text_input.value
        timestamp = dt.datetime.now()
        
        to_database = pd.DataFrame({'id': [query], 'as_expected': [relevant], 'sent_from': [1], 'comment': [comment],
                                    'datetimestamp': [timestamp]
                                   })
        to_database.to_sql('app_feedback', con=engine, chunksize=16000, if_exists='append', index=False)
        
        feedback_checkbox_1.value = False
        comment_textinput_1.value = '' 
    

# %%
feedback_checkbox_1 = pn.widgets.Checkbox(name='results relevant')
comment_textinput_1 = pn.widgets.TextEditor(placeholder='', height=200, width=300, toolbar=False, margin=5)
submit_feedback_button_1 = pn.widgets.Button(name='submit',  height=30, width=120,  margin=5, sizing_mode='fixed')
feedback_headline = pn.widgets.StaticText(name='feedback', 
                                    value='')
feedback_search = pn.Column(feedback_headline,
                            pn.Row(feedback_checkbox_1, submit_feedback_button_1, width=300), comment_textinput_1)

submit_feedback_button_1.on_click(submit_feedback_1)

# %%
tab_1 = pn.Column(text_search, pn.Row(text_input, 
                 pn.Column(delete_query_button, 
                           n_results_selector, 
                           pn.Spacer(width=10, height=20),
                           select_case_type,
                           pn.Spacer(width=10, height=20)),
                pn.Spacer(width=30),
                feedback_search
                                    ),
                 pn.Row(search_results_table, pn.Spacer(width=50))      
                 )

# %%
def return_empty_fig(title):
    fig = px.pie(pd.DataFrame({'a': [], 'b': []}), values='a', names='b')
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': title,
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        })
    return fig


def create_barchart(topics):
    if topics == []:
        return return_empty_fig('topics')
    
    topics = topics[0:12]
        
    subplot_titles = topics
   # colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])
        
    columns = 3
    rows = int(np.ceil(len(subplot_titles) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.5 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)
    fig.update_annotations(font_size=12)
    
    row = 1
    column = 1
    for i in topics:
        
        current = df_topics[df_topics['label'] == i]
        words = list(current['word'])
        scores = list(current['value'].fillna(0))
        
        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color= 'black' #next(colors)),
            ),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1
        
        
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"{'topics'}",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig
            
    
def create_doc_scatter(doc_df):
    fig = px.scatter(doc_df, x='x', y='y', color='topic', hover_data=['text'], size_max=20)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(hoverlabel=dict(font=dict(size=12)))
    
    return fig


def create_linechart(selection):
    if selection == []:
        df_ = pd.DataFrame({'date': [], 'values': []})
        fig = px.line(df_, x='date', y=df_.columns,
             hover_data={"date": "|%B %d, %Y"},
             title='topics time')
        fig.update_xaxes(dtick="M1",tickformat="%b\n%Y")
        return fig
    
    df_ = df[df.topic.isin(selection)][['CASE_OCCURENCE_DATE', 'topic']].copy()
    df_['counts'] = 1
    df_ = df_.groupby(['CASE_OCCURENCE_DATE', 'topic']).sum('counts').reset_index()
    df_['CASE_OCCURENCE_DATE'] = pd.to_datetime(df_['CASE_OCCURENCE_DATE'])
    r = pd.date_range(start=df_.CASE_OCCURENCE_DATE.min(), end=df_.CASE_OCCURENCE_DATE.max(), freq='D')
    
    
    df_ = df_.set_index(['topic', 'CASE_OCCURENCE_DATE'])['counts'].unstack().\
          reindex(columns=r,fill_value=0).stack().reset_index()
    df_ = df_.pivot(index='level_1', columns='topic', values=0)
    df_ = df_.fillna(0)
    
    df_['date'] = df_.index
    
    df_ = df_.groupby(pd.Grouper(key='date', axis=0, freq='M')).sum()
    df_['date'] = df_.index
    
    fig = px.line(df_, x='date', y=df_.columns,
             hover_data={"date": "|%B %d, %Y"},
              title='topics time')
    fig.update_xaxes(dtick="M1",tickformat="%b\n%Y")
    return fig


def create_doc_scatter(doc_df):
    fig = px.scatter(doc_df, x='x', y='y', color='topic', hover_data=['text'], size_max=20)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(hoverlabel=dict(font=dict(size=12)))
    
    return fig


def return_barchart_column(topics):
    if topics == []:
        return return_empty_fig('column by topic')
    
    
    topics = topics[0:12]
    
    selected_df = df[df.topic.isin(topics)].copy().groupby(['topic', column_selector.value])['CASENO'].count().reset_index()
    fig = px.bar(selected_df, x=column_selector.value, y='CASENO', color='topic', text_auto=True, barmode='group')
    fig.update_layout(
        title={
            'text': 'column by topic',
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        })

    return fig

def update_column_select(event):
    topics = list(topic_sel_table.value.iloc[topic_sel_table.selection]['topic'])
    barchart_columns.object = return_barchart_column(topics)
    
    
def update_topic_sel(value):
    current_sel = list(topic_sel_table.value.iloc[topic_sel_table.selection]['topic'])
    scatter_doc_df_  = scatter_doc_df[scatter_doc_df['topic'].isin(current_sel)].copy()
    fig = create_doc_scatter(scatter_doc_df_)
    doc_scatter.object = fig 
    
    barchart_words.object = create_barchart(current_sel)
    linechart_topics.object = create_linechart(current_sel)
    barchart_columns.object = return_barchart_column(current_sel)
    

    if current_sel != []:
        df_ = df[df.topic.isin(current_sel)].copy()
        df_['CASE_OCCURENCE_DATE'] = df_['CASE_OCCURENCE_DATE'].astype(str)
        df_['CASE_CLOSED_DATE'] = df_['CASE_CLOSED_DATE'].astype(str)
        topics_doc_table.header_filters = filters_panel_table
        topics_doc_table.value = df_
    else:
        topics_doc_table.header_filters = None
        topics_doc_table.value = return_empty_table()
        
    try:
        select_2.options = current_sel
        select_2.value = current_sel[0]
    except:
        pass
    
    
def submit_feedback_2(event):
    with pn.param.set_values(submit_feedback_button_2, loading=True):
        engine = create_engine(URL(
                account = account,
                user = user,
                password = password,
                database = database,
                schema = 'sinergi',
                role = role,
                warehouse = warehouse
            ))
        
        global edited_topics
        to_database = topic_sel_table.value.copy()
        to_database = to_database[to_database['topic'].isin(edited_topics)]
        to_database['datetimestamp'] = dt.datetime.now()
        to_database['sent_from'] = 2
        to_database['as_expected'] = True
        to_database = to_database.rename(columns={'topic': 'id', 'latest feedback': 'comment'})
        to_database = to_database[['id', 'as_expected','sent_from', 'comment', 'datetimestamp']]
        to_database.to_sql('app_feedback', con=engine, chunksize=16000, if_exists='append', index=False)
        edited_topics = []
        
        
def refresh_topics_feedback(event):
    with pn.param.set_values(reload_feedback_button_2, loading=True):
        engine = create_engine(URL(
                account = account,
                user = user,
                password = password,
                database = database,
                schema = 'sinergi',
                role = role,
                warehouse = warehouse
            ))
        
        sql = """select * from (select id, comment, datetimestamp from app_feedback where sent_from=2)
        qualify row_number() over (partition by id order by datetimestamp desc) = 1""" 
    
        feedback = pd.read_sql(sql, con=engine)
        global topic_counts
        topic_counts_ = pd.merge(topic_counts, feedback, how='left', left_on='topic', right_on='id')
        topic_counts_ = topic_counts_.fillna(' ')
        topic_counts_ = topic_counts_[['topic', 0, 'comment']]
        topic_counts_ = topic_counts_.rename(columns={'comment': 'latest feedback'})
        topic_sel_table.value = topic_counts_
        
        
def update_edited_rows(event):
    global edited_topics
    topic = topic_sel_table.value.loc[event.row]['topic']
    if topic not in edited_topics:
        edited_topics.append(topic) 
        

# %%
submit_feedback_button_2 = pn.widgets.Button(name='submit',  height=30, width=120,  margin=5, sizing_mode='fixed')
submit_feedback_button_2.on_click(submit_feedback_2)
reload_feedback_button_2 = pn.widgets.Button(name='refresh',  height=30, width=120,  margin=5, sizing_mode='fixed')
reload_feedback_button_2.on_click(refresh_topics_feedback)

# %%
edited_topics = []
texts = [fit_text(x, 120) for x in df['CASE_DESCRIPTION_EN'].values]
topics =  list(df['topic'].values)
scatter_doc_df = pd.DataFrame({'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1], 'topic': topics, 'text': texts})
fig = create_doc_scatter(pd.DataFrame({'x': [], 'y': [], 'topic': [], 'text': []}))
doc_scatter = pn.pane.Plotly(fig)

editors_topic_sel_table = {'topic': None, 0: None, 'latest feedback': TextEditor()}
topic_sel_table = pn.widgets.Tabulator(None,
                                       theme='simple', 
                                       show_index=False, 
                                       selectable='checkbox_single',
                                       editors=editors_topic_sel_table, 
                                       pagination=None, width=700,
                                       layout='fit_data_stretch',
                                       #widths={'topic': '35%', 'counts': '10%', 'comment': '55%'}, 
                                       text_align={'counts': 'left'})

topic_sel_table.on_edit(update_edited_rows)
topic_sel_table.param.watch(update_topic_sel, 'selection')
refresh_topics_feedback(None)

topics_doc_table = pn.widgets.Tabulator(return_empty_table(), theme='simple', 
                                        sizing_mode='stretch_width', 
                                        show_index=False, editors=editors_panel_table,
                                        selectable=False, layout='fit_columns',
                                        embed_content=False, 
                                        row_content=return_text, 
                                        pagination='local', 
                                        page_size=20,
                                        height=500, header_filters=None)

topics_doc_table.param.watch(update_expanded_table, 'expanded')

column_selector = pn.widgets.Select(name='Select Column', options=['FUNCTIONAL_AREA', 
                                                                   'EMPLOYMENT_CATEGORY', 
                                                                   'CASE_TYPE',  
                                                                   'CASE_SEVERITY',  
                                                                   'HAZARD_PHYSICAL_SECURITY_EVENT'], 
                                                                   value='CASE_TYPE',
                                    height=20, width=130, margin=2, sizing_mode='fixed')

column_selector.param.watch(update_column_select, 'value')

barchart_words = pn.pane.Plotly(create_barchart([]), width=500, height=400, sizing_mode='stretch_width')
barchart_columns = pn.pane.Plotly(return_barchart_column([]), width=500, height=400, sizing_mode='stretch_width')

linechart_topics = pn.pane.Plotly(create_linechart([]))

text_topics = pn.widgets.StaticText(name='Topic modeling', 
                                    value='''Topic modeling identifies topics in texts. Here, only the main topic per document is presented. <br />
                                     For the identification, we use a language model such that the topics do not depend on the actual vocabulary but on semantic similarity. <br /> 
                                     The representation text, on the other hand, is created using the actual documents.
Click on a topic and you can see the calculated similarity with other texts (topics graphic), the texts (topic table) or the occurrences over time (topics time).''')


# %%
tabs_documents = pn.Tabs(('topics graphic', doc_scatter), ('topics table', pn.Row(topics_doc_table, pn.Spacer(width=50))),
                ('topics time', linechart_topics), dynamic=True)
tabs_barcharts =  pn.Tabs(('topics words', barchart_words), ('column by topic', pn.Row(column_selector, barchart_columns)),
                           min_width=1000, sizing_mode='stretch_width', dynamic=True)

# %%
tab_2 = pn.Column(text_topics,pn.Row(pn.Column(pn.Row(submit_feedback_button_2, reload_feedback_button_2),
                                               topic_sel_table, height=400, scroll=True), tabs_barcharts), tabs_documents, 
                  pn.Spacer(width=50))

# %%
def update_date_range(event):
    start, end = date_range_slider.value
    recent_doc_df = df[(df['CASE_OCCURENCE_DATE'] >=start) & (df['CASE_OCCURENCE_DATE']  <= end)].copy()
    recent_doc_df = recent_doc_df.sort_values(by='CASE_OCCURENCE_DATE', ascending=False)
    recent_doc_table.value = recent_doc_df
    
    

def update_recent_neighbors(event):
    text = recent_doc_table.value.values[event.row, -1]
    global current_selection_text
    current_selection_text = text
    
    tensor = model.encode(text)
    cos_scores = util.cos_sim(tensor, embeddings_tensor)[0]
    vals, indices = torch.topk(cos_scores, k=n_results_recent_neighbors.value)
    
    case_nos = [embeddings_casenos[i] for i in indices]
    df_ = df.iloc[pd.Index(df['CASENO']).get_indexer(case_nos)].copy()
    
    topics = list(df_['topic'].values)
    df_['CASE_OCCURENCE_DATE'] = df_['CASE_OCCURENCE_DATE'].astype(str)
    df_['CASE_CLOSED_DATE'] = df_['CASE_CLOSED_DATE'].astype(str)
    
    recent_doc_neighbors_table.header_filters=filters_panel_table
    recent_doc_neighbors_table.value = df_
    
    linechart_recent_topics.object = create_linechart(topics)
    
    
def submit_feedback_3(event):
    with pn.param.set_values(feedback_recent_neighbors, loading=True):
        engine = create_engine(URL(
                account = account,
                user = user,
                password = password,
                database = database,
                schema = 'sinergi',
                role = role,
                warehouse = warehouse
            ))
            
        relevant = feedback_checkbox_3.value
        comment = comment_textinput_3.value
        query = current_selection_text
        timestamp = dt.datetime.now()
        
        to_database = pd.DataFrame({'id': [query], 'as_expected': [relevant], 'sent_from': [3], 'comment': [comment],
                                   'datetimestamp': [timestamp]
                                   })
        to_database.to_sql('app_feedback', con=engine, chunksize=16000, if_exists='append', index=False)
        
        feedback_checkbox_3.value = False
        comment_textinput_3.value = '' 
        
        

feedback_checkbox_3 = pn.widgets.Checkbox(name='results relevant')
comment_textinput_3 = pn.widgets.TextEditor(placeholder='', height=200, width=300, toolbar=False, margin=5)
submit_feedback_button_3 = pn.widgets.Button(name='submit',  height=30, width=120,  margin=5, sizing_mode='fixed')

feedback_recent_neighbors = pn.Column(pn.Row(feedback_checkbox_3, submit_feedback_button_3, width=300), comment_textinput_3)
submit_feedback_button_3.on_click(submit_feedback_3)


end = df['CASE_OCCURENCE_DATE'].max()
start = end - dt.timedelta(days=90)

date_range_slider = pn.widgets.DatetimeRangeSlider(
    name='date range',
    start=start, end=end, width=350,
    value=(end-dt.timedelta(days=7), end),
    step=24*3600*7*1000
)

date_range_slider.param.watch(update_date_range, 'value')

n_results_recent_neighbors = pn.widgets.Select(name='number results', options=list(range(10, 101, 10)), height=20, width=120, 
                                               value=20, sizing_mode='fixed', margin=5)
n_results_recent_neighbors.param.watch(update_recent_neighbors, 'value') 


current_selection_text = None
recent_doc_table = pn.widgets.Tabulator(return_empty_table(), theme='simple', 
                                        sizing_mode='stretch_width', 
                                        show_index=False, editors=editors_panel_table,
                                        selectable=1, layout='fit_data_stretch',
                                        embed_content=False, 
                                        row_content=return_text, 
                                        pagination='local', 
                                        page_size=10,
                                        height=400,
                                        width=None,
                                        margin=15,
                                        header_filters=filters_panel_table
                                       )


recent_doc_table.on_click(update_recent_neighbors)
recent_doc_table.param.watch(update_expanded_table, 'expanded')
update_date_range(None)


recent_doc_neighbors_table = pn.widgets.Tabulator(return_empty_table(), theme='simple', 
                                            sizing_mode='stretch_width', 
                                            show_index=False, editors=editors_panel_table,
                                            selectable=False, layout='fit_data_stretch',
                                            embed_content=False, 
                                            row_content=return_text, 
                                            pagination='local', 
                                            page_size=10,
                                            header_filters=None,
                                            height=400,
                                            width=None
                                           )

linechart_recent_topics = pn.pane.Plotly(create_linechart([]), height=400)


text_recent = pn.widgets.StaticText(name='Recent cases', 
                                    value=''' The top table presents new case. After selectiong one case, the lower tables shows semantic similar cases.''')



# %%
tab_3 = pn.Column(text_recent,date_range_slider, recent_doc_table, n_results_recent_neighbors,
                  pn.Tabs(('neighbors', recent_doc_neighbors_table), ('topics time', linechart_recent_topics), 
                          ('feedback', feedback_recent_neighbors), margin=35, dynamic=True))

# %%
tabs = pn.Tabs(('search', tab_1), ('topic modeling', tab_2), ('recent topics', tab_3), dynamic=True)

# %%
tabs.servable()

# %%



