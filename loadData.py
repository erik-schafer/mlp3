import xml.etree.ElementTree as ET
import json
import pandas as pd

class loadData:
    def loadSitecoreExtract():        
        tree = ET.parse("C:/Users/erik.schafer/Documents/sitecore_extract.xml")
        root = tree.getroot()

        #NLP Payload extraction
        nlpElements = root.findall("phrase[@fieldid='NLP Payload']")
        data = []
        for nlpItem in nlpElements:
            #nlpItem : ['path', 'key', 'itemid', 'fieldid', 'updated']
            nlpData = list(nlpItem)[0].text
            nlpData = json.loads(nlpData)
            #nlpData : ['usage', 'language', 'keywords', 'entities', 'emotion', 'concepts', 'categories']
            if 'keywords' in nlpData:
                keywords = nlpData['keywords']
                keywords = list(map(lambda x: (x['text'], x['relevance']), keywords))
            if 'entities' in nlpData:
                entities = nlpData['entities']
                entities = list(map(lambda x: (x['text'], x['relevance']), entities))
            if 'concepts' in nlpData:
                concepts = nlpData['concepts']
                concepts = list(map(lambda x: (x['text'], x['relevance']), concepts))
            if 'categories' in nlpData:
                categories = nlpData['categories']
                categories = list(map(lambda x: (x['label'], x['score']), categories))
            result = keywords + entities + concepts + categories
            #data.append({'id': nlpItem.get('itemid'), 'features': result})
            data.append(( nlpItem.get('itemid'),  result))

        #now we have a list of the following objects:
        ##{'id': <some guid>, 'features': [{<label>:<relevance>}]}
        # [("guid", [("term", .758),("term2", .213),...])]

        #count the occurences of all of the labels
        counts = {} #want [("term", 5), ("term2", 7), ...]
        for item in data:
            features = item[1]
            for k, v in features:
                if k not in counts:
                    counts[k] = 1
                else:
                    counts[k] = counts[k] + 1

        # below is some debug output
        #sortedDict = sorted(counts.items(), key = lambda x: x[1], reverse=True)
        #print(filter(lambda x: x[1] > 10, sortedDict))

        # let's drop all keys that have only a handful of content associated with them
        counts = {k: v for k, v in counts.items() if v > 50}

        # what we finally want is:
        # [id1, id2, id3, ... idn]
        # [[term1, term2, term3, ..., termn], [term1, term2, term3, ..., termn], ...]
        # [[relevance1, relevance2, .. relevancen], [relevance1, relevance2, .. relevancen], ...]
        dataKeys = counts.keys()
        ids = []
        dataValues = []
        for d in data:
            ids.append(d[0])
            dataValues.append([dict(d[1])[x] if x in dict(d[1]) else 0 for x in dataKeys])
            #dataValues.append([d['features'][x] if x in d['features'] else 0 for x in dataKeys])    
        print("end etl ")
        return pd.DataFrame(dataValues, columns = dataKeys)