import time
import datetime
import csv

from os.path import isfile
import untangle
import pandas as pd



def get_timestamp_from_event(event):
    date_time = ''
    if event is None:
        return None
    if  isinstance(event.date, list):
        for i in range(0, len(event.date)):
            if event.date[i]['key'] == u"time:timestamp":
                date_time = event.date[i]['value']
    else:
        date_time = event.date['value']

    timestamp = time.mktime(datetime.datetime.strptime(
        date_time[0:19], "%Y-%m-%dT%H:%M:%S").timetuple())

    return timestamp


def encode(fileName, prefix):
    if isfile('core_encodedFiles/boolean_' + fileName + '_' + str(prefix) + '.csv'):
        return None

    obj = untangle.parse('logdata/' + fileName)
    attrs_dictionary = {}
    unique_events_resource = list()

    bool_encoded_traces = list()
    freq_encoded_traces = list()
    complex_index_traces = list()
    latest_payload_index_traces = list()
    simple_index_traces = list()

    event_attr_names = list()
    event_attr_names_header = list()
    event_attr_latest_names = list()
    events_header = list()
    key_value = 1

    print "============================================="
    print obj.log.trace[0].event[0].children[0]['key']
    lenOfAttr = len(obj.log.trace[0].event[0].children)

    for event_attr in obj.log.trace[0].event[prefix - 1].children:
        event_attr_latest_names.append("Latest_" + event_attr['key'])
        event_attr_names.append(event_attr['key'])

    for i in xrange(1, prefix + 1):
        events_header.append("Event_" + str(i))
        for attr in event_attr_names:
            event_attr_names_header.append(attr + "_" + str(i))

    for trace in obj.log.trace:
        if len(trace.event) > prefix:
            for event in trace.event[0:prefix]:
                if event.string[2]['value'] not in unique_events_resource:
                    unique_events_resource.append(event.string[2]['value'])

    for trace in obj.log.trace:
        if len(trace.event) >= prefix:
            bool_trace = list()
            freq_trace = list()
            trace_event_ids = list()
            if type(trace.string) is list:
                for i in range(0, len(trace.string)):
                    if u"concept:name" == trace.string[i]['key']:
                        trace_case_id = trace.string[i]['value']
            else:
                # only has 1 value, so it automatically becomes the case id
                trace_case_id = trace.string['value']
            
            
            #trace_case_id = trace.string[0]['value']

            bool_trace.append(trace_case_id)
            freq_trace.append(trace_case_id)
            trace_events = list()

            first_event_timestamp_ = get_timestamp_from_event(
                trace.event[0])
            last_event_timestamp_ = get_timestamp_from_event(
                trace.event[len(trace.event) - 1])
            duration = last_event_timestamp_ - first_event_timestamp_
            last_prefix_remainingTime = get_timestamp_from_event(
                trace.event[prefix - 1])
            last_prefix_event_attr = list()
            event_attr_values = list()

            for event_attr in event_attr_names:
                foundValue_ = False;
                value_ = 0;
                for chlid in trace.event[prefix - 1].children:
                    if event_attr == chlid['key']:
                        value_ = chlid['value']
                        foundValue_ = True;
                if(foundValue_):
                    if value_ not in attrs_dictionary:
                        attrs_dictionary[value_] = key_value
                        key_value = key_value + 1
                        
                    last_prefix_event_attr.append(attrs_dictionary[value_])
                else: 
                    last_prefix_event_attr.append(0)



            for event in trace.event[0:prefix]:
                event_case_id = "event_case_id"
                if type(event.string) is list:
                    for i in range(0, len(event.string)):
                        if u"concept:name" == event.string[i]['key']:
                            event_case_id = event.string[i]['value']

                if event_case_id not in attrs_dictionary:
                    attrs_dictionary[event_case_id] = key_value
                    key_value = key_value + 1
                trace_event_ids.append(
                    attrs_dictionary[event_case_id])
                for event_attr in event_attr_names:
                    foundValue = False;
                    value = 0;
                    for chlid in event.children:
                        if event_attr == chlid['key']:
                            value = chlid['value']
                            foundValue = True;
                    if(foundValue):
                        if value not in attrs_dictionary:
                            attrs_dictionary[value] = key_value
                            key_value = key_value + 1
                        event_attr_values.append(attrs_dictionary[value])
                    else: 
                        event_attr_values.append(0)
        
                trace_events.append(event.string[2]['value'])
            complex_index_traces.append(
                [trace_case_id] + event_attr_values + [last_event_timestamp_ - last_prefix_remainingTime] + [duration])
            latest_payload_index_traces.append([trace_case_id] + trace_event_ids + last_prefix_event_attr + [
                                               last_event_timestamp_ - last_prefix_remainingTime] + [duration])
            simple_index_traces.append(
                [trace_case_id] + trace_event_ids + [last_event_timestamp_ - last_prefix_remainingTime] + [duration])

            # Bool and Freq encoding
            for unique_id in unique_events_resource:
                if (unique_id in trace_events):
                    bool_trace.append("1")
                    freq_trace.append(trace_events.count(unique_id))
                else:
                    bool_trace.append("0")
                    freq_trace.append("0")
            bool_encoded_traces.append(
                bool_trace + [last_event_timestamp_ - last_prefix_remainingTime] + [duration])
            freq_encoded_traces.append(
                freq_trace + [last_event_timestamp_ - last_prefix_remainingTime] + [duration])

    with open('core_encodedFiles/boolean_' + fileName + '_' + str(prefix) + '.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Id"] + unique_events_resource + ["remainingTime"] + ['duration'])
        for x in bool_encoded_traces:
            wr.writerow(x)

    with open('core_encodedFiles/frequency_' + fileName + '_' + str(prefix) + '.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Id"] + unique_events_resource + ["remainingTime"] + ['duration'])
        for x in freq_encoded_traces:
            wr.writerow(x)

    with open('core_encodedFiles/complexIndex_' + fileName + '_' + str(prefix) + '.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Id"] + event_attr_names_header + ["remainingTime"] + ['duration'])
        for x in complex_index_traces:
            wr.writerow(x)

    with open('core_encodedFiles/simpleIndex_' + fileName + '_' + str(prefix) + '.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Id"] + events_header + ["remainingTime"] + ['duration'])
        for x in simple_index_traces:
            wr.writerow(x)

    with open('core_encodedFiles/indexLatestPayload_' + fileName + '_' + str(prefix) + '.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Id"] + events_header +
                    event_attr_latest_names + ["remainingTime"] + ['duration'])
        for x in latest_payload_index_traces:
            wr.writerow(x)

    with open('core_encodedFiles/encoding_dictionary_' + fileName + '_' + str(prefix) + '.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for key, value in attrs_dictionary.items():
            wr.writerow([key, value])

def fast_slow_encode(fileName, prefix, encoding, label, threshold):
    filename = 'core_encodedFiles/'+ encoding +'_' + fileName + '_' + str(prefix) + '.csv'
    if isfile(filename):
        df = pd.read_csv(filename)
        if threshold == "default":
            threshold_ = df[label].mean()
            print threshold_
        else:
            threshold_ = float(threshold)

        
        df['actual'] = df[label] < threshold_
        # df = df.sample(frac=1)


        return df
    return None
# bool_db = cluster.DBSCAN(eps=2,min_samples=5)
# 	bool_db.fit(bool_encoded_traces)

# 	print bool_db

# 	bool_kmeans = cluster.KMeans(n_clusters=5)

# 	clustered_data = list()
# 	labels = bool_kmeans.labels_
# 	# for i in range (0, len()):


# 	bool_kmeans.fit(bool_encoded_traces)
# 	print bool_kmeans[0]

# 	print bool_kmeans.cluster_centers_
