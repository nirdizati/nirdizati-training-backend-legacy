import json
import logging
import os
import sys
from datetime import datetime as d_time
from math import sqrt

import numpy as np
import pandas as pio
import xgboost as xgb
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Get an instance of a logger
logger = logging.getLogger("testing")

def freq_encoding(df, state_list):

    cur_freq = []

    for s in state_list:
        name_ = 'state_freq' + '_' + str(s)
        df[name_] = 0

    cur_case = -1

    for i in range(0, len(df)):
        if cur_case != df.at[i, 'id']:
            cur_freq = [0] * len(state_list)
        else:
            try:
                ind = state_list.index(df.at[i, 'state'])
                cur_freq[ind] += 1
                df.at[i, 'state_freq' + '_' + str(df.at[i, 'state'])] = cur_freq[ind]
            except ValueError:
                print 'err'
                return ValueError
        cur_case = df.at[i, 'id']
    return df



def get_history_len(df):
    max_size = -1
    for i in range(0, len(df)):
        if str(df.at[i, 'history']) != "nan":
            parsed = df.at[i, 'history'].split("_")
            if len(parsed) > max_size:
                max_size = len(parsed)
    return max_size


def history_encoding_new(df):
    hist_len = get_history_len(df)

    for k in range(0, hist_len):
        df['event_' + str(k)] = -1
    for i in range(0, len(df)):
        if str(df.at[i, 'history']) != "nan":
            parsed_hist = str(df.at[i, 'history']).split("_")
            for k in range(0, len(parsed_hist)):
                df.at[i, 'event_' + str(k)] = int(parsed_hist[k])

    return df, hist_len


def prep_data(df, state_list, query_name, level):

    cols = ['resource']

    df, hist_len = history_encoding_new(df)

    for h in range(0,hist_len):
     cols.append('event_'+str(h))

    for c in cols:
        df[c] = df[c].astype('category')

    df_categorical = df[cols]

    dummies = pio.get_dummies(df_categorical)
    cols = ['elapsed_time',query_name]

    if level == 'Level3':
        for k,s in enumerate(state_list):
           cols.append('pref'+'_'+str(k))
    elif level == 'Level1':
        for k,s in enumerate(state_list):
           cols.append('queue'+'_'+str(k))

    df_numerical = df[cols]
    df_numerical = pio.concat([df_numerical, dummies], axis=1)
    train_df, test_df = train_test_split(df_numerical, test_size=0.2, random_state=3)
    print train_df.head(3)

    train_list = train_df.index.values.tolist()
    test_list = test_df.index.values.tolist()
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df, train_list, test_list

def Lasso_Regression(train_df, test_df):
        #alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10]
        LR = LassoCV(fit_intercept=True, normalize=True, n_jobs=-1, max_iter=100000, tol=0.001)
        X_train = train_df.ix[:, train_df.columns != 'remaining_time']
        X_test = test_df.ix[:, test_df.columns != 'remaining_time']
        # feat_list = ['elapsed_time','age','simple_snapshot','state','staff','gender','arrival']
        y = train_df['remaining_time']
        LR.fit(X_train, y)
        y_test = test_df['remaining_time']
        # prediction, bias, contributions = ti.predict(rf, X_test)
        # assert (numpy.allclose(prediction, bias + np.sum(contributions, axis=1)))
        # assert (numpy.allclose(rf.predict(X_test), bias + np.sum(contributions, axis=1)))
        print LR.alpha_
        y_pred = LR.predict(X_test)
        rms = sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print 'Lasso RMSE = ' + str(round(rms / 60, 2))
        print 'Lasso MAE = ' + str(round(mae / 60, 2))
        return y_pred

def random_Forest_regression(train_df, test_df, n_est, sample_leaf, query_name):
    rf = RandomForestRegressor(n_estimators=n_est, n_jobs=8, verbose=1)  # ,
    # oob_score = True)
    X_train = train_df.ix[:, train_df.columns != query_name]
    X_test = test_df.ix[:, test_df.columns != query_name]
    # feat_list = ['elapsed_time','age','simple_snapshot','state','staff','gender','arrival']
    y = train_df[query_name]
    rf.fit(X_train, y)
    y_test = test_df[query_name]
    # prediction, bias, contributions = ti.predict(rf, X_test)
    # assert (numpy.allclose(prediction, bias + np.sum(contributions, axis=1)))
    # assert (numpy.allclose(rf.predict(X_test), bias + np.sum(contributions, axis=1)))
    print rf
    y_pred = rf.predict(X_test)
    y_pred[y_pred < 0] = 0
    rms = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print 'Random Forest RMSE = ' + str(round(rms / 60, 2))
    print 'Random Forest MAE = ' + str(round(mae / 60, 2))
    return y_pred

def XG_Boosting_Regression(train_df, test_df, n_est, md, query_name):
    X_train = train_df.ix[:, train_df.columns != query_name]
    X_test = test_df.ix[:, test_df.columns != query_name]

    # feat_list = ['elapsed_time','age','simple_snapshot','state','staff','gender','arrival']
    y = train_df[query_name]
    # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #   'learning_rate': 0.01, 'loss': 'ls'}
    clf = xgb.XGBRegressor(n_estimators=n_est, max_depth=md)  # , verbose = 1)
    # clf = GridSearchCV(xgb_model,
    #                {'max_depth': [6,8,10],
    #               'n_estimators': [200,250,300]}, verbose=1)

    # clf.fit(X_train, y)
    clf.fit(X_train, y)

    #xgb.plot_importance(clf)
    y_test = test_df[query_name]

    y_pred = clf.predict(X_test)
    y_pred[y_pred < 0] = 0
    rms = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print 'XGB RMSE = ' + str(round(rms / 60, 2))
    print 'XGB MAE = ' + str(round(mae / 60, 2))

    # plot_deviance(clf,y_test,X_test,n_est)
    return y_pred


def XG_Boosting_Regression_CV(train_df, test_df, n_est, md, query_name):
    # cross validating XGBoost
    train_df_new, eval_df = train_test_split(train_df, test_size=0.25, random_state=3)
    X_train = train_df_new.ix[:, train_df.columns != query_name]
    X_eval = eval_df.ix[:, train_df.columns != query_name]
    X_test = test_df.ix[:, test_df.columns != query_name]
    # feat_list = ['elapsed_time','age','simple_snapshot','state','staff','gender','arrival']
    y = train_df_new[query_name]
    y_eval = eval_df[query_name]
    # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #   'learning_rate': 0.01, 'loss': 'ls'}
    clf = xgb.XGBRegressor(max_depth=md, n_estimators=n_est)  # , verbose = 1)

    eval_list = [(X_eval, y_eval)]
    clf.fit(X_train, y, eval_set=eval_list, verbose=True, early_stopping_rounds=100)
    #xgb.plot_importance(clf)
    y_test = test_df[query_name]
    y_pred = clf.predict(X_test)
    y_pred[y_pred < 0] = 0
    rms = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print 'XGB RMSE = ' + str(round(rms / 60, 2))
    print 'XGB MAE = ' + str(round(mae / 60, 2))

    # plot_deviance(clf,y_test,X_test,n_est)
    return y_pred


def linear_regression(train_df, test_df, query_name):
    lm = LinearRegression(fit_intercept=True)
    X_train = train_df.ix[:, train_df.columns != query_name]
    X_test = test_df.ix[:, test_df.columns != query_name]
    # feat_list = ['elapsed_time','age','simple_snapshot','state','staff','gender','arrival']
    y = train_df[query_name]
    lm.fit(X_train, y)
    print lm
    y_pred = lm.predict(X_test)
    y_pred[y_pred < 0] = 0
    y_test = test_df[query_name]

    rms = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print 'Linear Regression RMSE = ' + str(round(rms / 60, 2))
    print 'Linear Regression MAE = ' + str(round(mae / 60, 2))
    return y_pred



def plot_results(df, test_list,state_list):
    plot_df = df.ix[test_list]
    cols = ['id', 'elapsed_time', 'total_time', 'remaining_time']
    plot_df = plot_df[cols]
    return plot_df


def ML_methods(df, state_list, query_name, filename, level):
    train_df, test_df, train_list, test_list = prep_data(df, state_list, query_name, level)
    print "Training shape is: " + str(train_df.shape)

    print "Fitting Random Forest"
    trees = [50]
    #trees = []
    rf_results = []
    for t in trees:
        print "Number of trees: " + str(t)
        rf_results.append(random_Forest_regression(train_df, test_df, t, 2, query_name))
    print "Fitting XGBOOST"
    trees_xg = [2000]
    xg_results = []
    # used to be md = 100
    for t in trees_xg:
        print "Number of trees: " + str(t)
        #xg_results.append(XG_Boosting_Regression_CV(train_df, test_df, t, 10, query_name))
        xg_results.append(XG_Boosting_Regression(train_df, test_df, t, 10, query_name))
    print "Fitting Lasso CV"
    y_lasso = Lasso_Regression(train_df, test_df)
    # print "Fitting Linear Regression"
    # y_linear = linear_regression(train_df, test_df, query_name)

    plot_df = plot_results(df, test_list,state_list)

    for j,y in enumerate(xg_results):
        method = 'XG_' + str(trees_xg[j])
        plot_df[method] = y
    for j,y in enumerate(rf_results):
        method = 'RF_' + str(trees[j])
        plot_df[method] = y
    # method = 'LM_pred'
    # plot_df[method] = y_linear
    method = 'Lasso'
    plot_df[method] = y_lasso
    # Version includes: query + baseline + qmethod list
    V = level+filename
    print 'saving file: ' + V
    results_filename = write_pandas_to_csv(plot_df,V,out = True)

    return results_filename


def read_into_panda_from_csv(path):

    panda_log = pio.read_csv(filepath_or_buffer=path, header=0)#, index_col=0)
    # rename columns:
    panda_log.columns = ['id','timestamp','state','resource']
    panda_log = panda_log.sort(['id', 'timestamp'], ascending=True)
    panda_log = panda_log.reset_index(drop=True)

    return panda_log


def add_next_state(df):
    df['next_state'] = ''
    df['next_time'] = 0
    df['next_dur'] = 0
    num_rows = len(df)
    for i in range(0, num_rows - 1):
        print str(i) + ' out of ' + str(num_rows)

        if df.at[i, 'id'] == df.at[i + 1, 'id']:
            df.at[i, 'next_state'] = df.at[i + 1, 'state']
            df.at[i, 'next_time'] = df.at[i + 1, 'timestamp']
            df.at[i, 'next_dur'] = df.at[i + 1, 'timestamp'] - df.at[i, 'timestamp']
        else:
            df.at[i, 'next_state'] = 99
            df.at[i, 'next_time'] = df.at[i, 'timestamp']
            df.at[i, 'next_dur'] = 0
    df.at[num_rows-1, 'next_state'] = 99
    df.at[num_rows-1, 'next_time'] = df.at[num_rows-1, 'timestamp']
    df.at[num_rows-1, 'next_dur'] = 0

    return df


def add_query_remaining(df):
    df['elapsed_time'] = 0
    df['total_time'] = 0
    df['remaining_time'] = 0
    df['history'] = ""
    ids = []
    total_Times = []
    num_rows = len(df)
    temp_elapsed = 0
    prefix = str(df.at[0, 'state'])
    df.at[0, 'history'] = prefix

    for i in range(1, num_rows):
        # print i
        print str(i) + ' out of ' + str(num_rows)

        if df.at[i, 'id'] == df.at[i - 1, 'id']:
            temp_elapsed += df.at[i - 1, 'next_dur']
            df.at[i, 'elapsed_time'] = temp_elapsed
            prefix = prefix + '_' + str(df.at[i, 'state'])
            df.at[i, 'history'] = prefix
        else:
            ids.append(df.at[i - 1, 'id'])
            total_Times.append(temp_elapsed)
            temp_elapsed = 0
            prefix = str(df.at[i, 'state'])
            df.at[i, 'history'] = prefix

    ids.append(df.at[num_rows - 1, 'id'])
    total_Times.append(df.at[num_rows - 1, 'elapsed_time'])
    # df.at[num_rows-1,'elapsed_time'] = temp_elapsed
    for i in range(0, num_rows):
        print str(i) + ' out of ' + str(num_rows)
        try:
            ind = ids.index(df.at[i, 'id'])
            total_ = total_Times[ind]
            df.at[i, 'total_time'] = total_
            df.at[i, 'remaining_time'] = total_ - df.at[i, 'elapsed_time']
            # print df.head(i)
        except ValueError:
            print 'err'
            return ValueError
    return


def is_empty(any_structure):
    if any_structure:
        print('Structure is not empty.')
        return False
    else:
        print('Structure is empty.')
        return True


# def encode_sequence(df):
def write_pandas_to_csv(df, version, out):
    # df = df.reset_index(drop=True)
    filename = ''
    if out == False:
        filename = 'Query_Remaining_Time' + str(version) + '.csv'
        df.to_csv(filename,sep=',')
    else:
        filename = 'Results/' + str(version) + '.csv'
        df.to_csv('Results/' + str(version) + '.csv', sep=',')
    return filename


def create_initial_log(path, name):
    df = read_into_panda_from_csv(path)

    add_next_state(df)
    add_query_remaining(df)
    #df = clean_outliers(df)
    os.path.splitext(path)[0]
    version = "_level_0"+name
    filename = write_pandas_to_csv(df, version, False)
    return filename

def order_csv_time(path, name):
    df = pio.read_csv(filepath_or_buffer=path, header=0, index_col=0)  # , nrows= 20)
    df = df.sort(['timestamp'], ascending=True)
    df = df.reset_index(drop=True)
    version = "V_events_0_ordered"+name
    filename = write_pandas_to_csv(df, version, False)
    return filename

def read_from_query(path):
    df = pio.read_csv(filepath_or_buffer=path, header=0, index_col=0)  # ,nrows = 1000)
     # List = range(0,len(df))
    # df = df.ix[List]

    return df


def add_hour(df):
    df['hr'] = 0
    for i in range(0, len(df)):
        if pio.isnull(df.at[i, 'next_state']):
            df.at[i, 'next_state'] = 22
            if df.at[i,'next_time'] == 0:
                df.at[i,'next_time']= df.at[i,'timestamp']
        try:
            df.at[i, 'hr'] = d_time.strptime(df.at[i, 'date_time'], "%d/%m/%Y %H:%M").hour
        except ValueError:
            df.at[i, 'hr'] = d_time.strptime(df.at[i, 'date_time'], "%d/%m/%Y").hour
        except TypeError:
            print 'Hr err'
    return


def get_states(df):
    state_list = []
    for i in range(0, len(df)):
        pair = df.at[i, 'state']
        try:
            ind = state_list.index(pair)
        except ValueError:
            state_list.append(pair)
    return state_list


def update_event_queue(event_queue, cur_time):
    remove_indices = []
    rem_ind = []

    # going over the different states and getting the rates
    for i, e in enumerate(event_queue):
        for j, q in enumerate(event_queue[i]):
            if q[1] <= cur_time:
                rem_ind.append(j)
        remove_indices.append(rem_ind)

        # print 'count remove: ' + str(count_remove)
        count_remove = 0
        if len(remove_indices[i]) > 0:
            for index in sorted(remove_indices[i], reverse=True):
                del event_queue[i][index]
        rem_ind = []
    return

def find_q_len_ttiq(event_queue, cur_time):
    q_len = len(event_queue)
    return q_len
def find_mc_q(event_queue, cur_time):
    q_len = len(event_queue)
    return q_len
def add_queues(df, state_list):
    event_queue = []
    tuple = []
    df['total_q'] = 0

    for s in state_list:
        col_name = 'queue' + '_' + str(s)
        df[col_name] = 0
        event_queue.append(tuple)
        tuple = []

    num_rows = len(df)
    for i in range(0, num_rows):
        # print (str(i) + ' queueing calculation')

        cur_time = df.at[i, 'timestamp']
        next_time = df.at[i, 'next_time']
        cur_state = df.at[i, 'state']
        ind = state_list.index(cur_state)
        tuple = [cur_time, next_time]
        event_queue[ind].append(tuple)
        update_event_queue(event_queue, cur_time)


        total_q = 0
        for j, s in enumerate(state_list):
            col_name1 = 'queue' + '_' + str(s)
            ind = state_list.index(s)
            x = find_q_len_ttiq(event_queue[ind], cur_time)
            df.at[i, col_name1] = x
            total_q += x
        df.at[i,'total_q'] = total_q

    return df
def add_mc_queues(df, pref_list):
    event_queue = []
    tuple = []
    recent_occur = []
    delta = []
    print "Number of prefixes is "+str(len(pref_list))
    for k,s in enumerate(pref_list):
        col_name = 'pref' + '_' + str(k)
        print str(s)
        df[col_name] = 0
        event_queue.append(tuple)
        tuple = []

    num_rows = len(df)
    for i in range(0, num_rows):
        # print (str(i) + ' queueing calculation')
        # cur_state = r.state.values[0]
        cur_time = df.at[i, 'timestamp']
        next_time = df.at[i, 'next_time']
        #cur_state = df.at[i, 'state']
        memorylen= 3
        hist = df.at[i, 'history']
        parsed_hist = str(hist).split("_")
        if len(parsed_hist)>memorylen:

            #History is too long.
            hist = ''
            for k in range(0, len(parsed_hist)):
                if k > memorylen-1:
                    break
                else:
                    if hist=='':
                        hist = hist + str(parsed_hist[len(parsed_hist) - k - 1])
                    else:
                        hist = str(parsed_hist[len(parsed_hist) - k - 1])+'_'+ hist

        ind = pref_list.index(hist)
        tuple = [cur_time, next_time]
        event_queue[ind].append(tuple)
        update_event_queue(event_queue, cur_time)


        for j, s in enumerate(pref_list):
            col_name1 = 'pref' + '_' + str(j)
            ind = pref_list.index(s)
            df.at[i, col_name1] = find_mc_q(event_queue[ind], cur_time)
    return df


def queue_level(path_query, name):

    df = read_from_query(path_query)
    df = df.reset_index(drop=True)
    state_list = get_states(df)
    df = add_queues(df, state_list)
    version = "V_events_3"+name
    filename = write_pandas_to_csv(df, version, False)
    return filename

def get_prefixes(df):
    memorylen = 3
    pref_list = []
    for i in range(0, len(df)):
        hist = df.at[i, 'history']
        parsed_hist = str(hist).split("_")
        if len(parsed_hist)<=memorylen:

            try:
                ind = pref_list.index(hist)
            except ValueError:
                pref_list.append(hist)
        else:
            #History is too long.
            hist = ''
            for k in range(0, len(parsed_hist)):
                if k > memorylen-1:
                    break
                else:
                    if hist=='':
                        hist = hist + str(parsed_hist[len(parsed_hist) - k - 1])
                    else:
                        hist = str(parsed_hist[len(parsed_hist) - k - 1])+'_'+ hist

            try:
                ind = pref_list.index(hist)
            except ValueError:
                pref_list.append(hist)
    return pref_list

def multiclass(path_query, name):
    df = read_from_query(path_query)
    df = df.reset_index(drop=True)
    pref_list = get_prefixes(df)
    df = add_mc_queues(df, pref_list)
    version = "V_events_4"+name
    filename = write_pandas_to_csv(df, version, False)
    return filename

def handle_uploaded_file(f):
    print 'handle file upload'
    with open('sample.csv', 'wb+') as destination:
        for chunk in f.chunks():
            print chunk
            print 'hello world'
            destination.write(chunk)

@csrf_exempt
def index(request):
    response_data = {}
    response_data['result'] = 'error'
    response_data['message'] = 'No request'

    print request.method
    if request.method == 'POST':
        req = json.loads(request.body)
        level = req['level']
        filename = req['name']
        print >> sys.stderr, 'POST detected'
        # handle_uploaded_file('prepdata.csv')
        # filename = request.FILES['file'].name
        # filename = json.loads(request.body.decode('utf-8'))['name']
        name = filename
        filename = 'logdata/'+filename+'.csv'

        # encode the file -- level 0
        level0_file = create_initial_log(filename ,name)
        # encode the file -- level 0 order file
        level0_file_ordered = order_csv_time(level0_file, name)

        #make the predictions
        df = {}
        state_list = {}
        query_name = 'remaining_time'
        if level == 'Level3':
            # # encode the file -- level 3
            level3_file = multiclass(level0_file_ordered, name)
            df = read_from_query(level3_file)
            state_list = get_prefixes(df)
        elif level == 'Level2' or level == 'Level1':
            # # encode the file -- level 1 and 2
            level1_2_file = queue_level(level0_file_ordered, name)
            df = read_from_query(level1_2_file)
            state_list = get_states(df)
        elif level == 'Level0':
            df = read_from_query(level0_file_ordered)

        print df.head(20)

        results_filename = ML_methods(df, state_list, query_name, name, level)

        results = {}
        results["message"] = "results in " + results_filename
        # results["data"] = read_from_query(results_filename)
        return HttpResponse(json.dumps(results), content_type="application/json")
    else:
        print 'here response'
        return HttpResponse(json.dumps(response_data), content_type="application/json")

@csrf_exempt
def results(request):
    response_data = {}
    filename = request.GET['file']
    df = read_from_query("Results/"+filename)
    df['remaining_time'] = df['remaining_time']/3600
    df['Lasso'] = df['Lasso'] / 3600
    # df['LM_pred'] = df['LM_pred'] / 3600
    df['XG_2000'] = df['XG_2000'] / 3600
    df['RF_50'] = df['RF_50'] / 3600
    return HttpResponse(df.to_json(), content_type="application/json")

def root_mean_square_error_calculation(df, field1, field2):
    return sqrt(mean_squared_error(df[field1], df[field2]))

def mean_absolute_error_calculation(df, field1, field2):
    return mean_absolute_error(df[field1], df[field2])

@csrf_exempt
def get_general_evaluation(request):
    response_data = {}
    filename = request.GET['file']
    df = read_from_query("Results/"+filename)
    df['remaining_time'] = df['remaining_time']/3600
    df['Lasso'] = df['Lasso'] / 3600
    # df['LM_pred'] = df['LM_pred'] / 3600
    df['XG_2000'] = df['XG_2000'] / 3600
    df['RF_50'] = df['RF_50'] / 3600
    results = {}
    rmse = {}
    rmse['Lasso'] = root_mean_square_error_calculation(df, 'remaining_time', 'Lasso')
    # rmse['LM_pred'] = root_mean_square_error_calculation(df, 'remaining_time', 'LM_pred')
    rmse['RF_50'] = root_mean_square_error_calculation(df, 'remaining_time', 'RF_50')
    rmse['XG_2000'] = root_mean_square_error_calculation(df, 'remaining_time', 'XG_2000')
    results['RMSE'] = rmse

    mae = {}
    mae['Lasso'] = mean_absolute_error_calculation(df, 'remaining_time', 'Lasso')
    # mae['LM_pred'] = mean_absolute_error_calculation(df, 'remaining_time', 'LM_pred')
    mae['RF_50'] = mean_absolute_error_calculation(df, 'remaining_time', 'RF_50')
    mae['XG_2000'] = mean_absolute_error_calculation(df, 'remaining_time', 'XG_2000')
    results['MAE'] = mae

    return HttpResponse(json.dumps(results), content_type="application/json")

@csrf_exempt
def get_evaluation(request):
    response_data = {}
    filename = request.GET['file']
    df = read_from_query("Results/"+filename)

    df = df.sort('remaining_time', ascending=False)
    df = df.reset_index(drop=True)

    #convert to hours
    df['remaining_time'] = df['remaining_time']/3600
    df['Lasso'] = df['Lasso'] / 3600
    # df['LM_pred'] = df['LM_pred'] / 3600
    df['XG_2000'] = df['XG_2000'] / 3600
    df['RF_50'] = df['RF_50'] / 3600

    range = {}
    range_list = {}
    remaining_time = df['remaining_time']
    divider = 20

    range_difference = np.amax(remaining_time) / divider
    x = 0
    counter = 20
    while x < 20:
        counter -= 1
        range_start = (counter+1)*range_difference -1
        range_end = counter * range_difference
        range_string = "%s - %s" %(range_start, range_end)
        #filter data
        range_data = df[df['remaining_time'] <= range_start ]
        range_data = range_data[range_data['remaining_time'] >= range_end]

        data = {}
        rmse = {}

        if len(range_data) == 0:
            # range[x] = {}
            # range_list[x] = range_string
            x += 1
            continue

        rmse['Lasso'] = root_mean_square_error_calculation(range_data, 'remaining_time', 'Lasso')
        # rmse['LM_pred'] = root_mean_square_error_calculation(range_data, 'remaining_time', 'LM_pred')
        rmse['RF_50'] = root_mean_square_error_calculation(range_data, 'remaining_time', 'RF_50')
        rmse['XG_2000'] = root_mean_square_error_calculation(range_data, 'remaining_time', 'XG_2000')
        data['RMSE'] = rmse

        mae = {}
        mae['Lasso'] = mean_absolute_error_calculation(range_data, 'remaining_time', 'Lasso')
        # mae['LM_pred'] = mean_absolute_error_calculation(range_data, 'remaining_time', 'LM_pred')
        mae['RF_50'] = mean_absolute_error_calculation(range_data, 'remaining_time', 'RF_50')
        mae['XG_2000'] = mean_absolute_error_calculation(range_data, 'remaining_time', 'XG_2000')
        data['MAE'] = mae

        range[x] = data
        range_list[x] = range_string
        x += 1

    results = {}
    results['data'] = range
    results['intervals'] = range_list
    return HttpResponse(json.dumps(results), content_type="application/json")