"""
Exploring David Mezzetti great intro article at:
<https://medium.com/neuml/getting-started-with-semantic-search-a9fd9d8a48cf>.
"""

import pprint
from txtai.scoring import ScoringFactory


print( ' ')
print( '-' * 70 )
print( '(1) Starting data...' )
print( '-' * 70 )

data = ["US tops 5 million confirmed virus cases",
        "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
        "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
        "The National Park Service warns against sacrificing slower friends in a bear attack",
        "Maine man wins $1M from $25 lottery ticket",
        "Make huge profits without work, earn up to $100,000 a day"]

pprint.pprint( data )
print("-" * 70)
print( ' ')

print("-" * 70)
print( '(2) Building SOLR-like keyword index...')


## Create a BM25 index (solr-like) ----------------------------------
scoring = ScoringFactory.create({"method": "bm25", "terms": True})
assert repr( type(scoring) ) == "<class 'txtai.scoring.bm25.BM25'>"
if scoring:  # just to get rid of type-check error
    scoring.index( ((x, text, None) for x, text in enumerate(data)) )
# print( f'scoring.index, ``{pprint.pformat(scoring.__dict__)}``' )

print( '...Index built')
print("-" * 70)
print( ' ')

print( '-' * 70 )
print( '(3) Showing good keyword query results on keyword index...' )
print( '-' * 70 )
print( ' ')

print("%-20s %s" % ("Query", "Best Match"))
print("-" * 50)

## run keyword solr-like query successfully -------------------------
for query in ("lottery winner", 
              "canadian iceberg", 
              "number of cases",
              "rising tensions", 
              "park service"):
    ## Get index of best section that best matches query
    if scoring:
        results = scoring.search(query, 1)
        # print( f'results, ``{results}``' )
        match = data[results[0][0]] if results else "No results"
        print("%-20s %s" % (query, match))

print( ' ')
print( '-' * 70 )
print( '(4) Showing bad `semantic` query results on keyword index...' )
print( '-' * 70 )
print( ' ')

for query in ("feel good story", 
              "climate change", 
              "public health story",
              "war", 
              "wildlife", 
              "asia", 
              "lucky", 
              "dishonest junk"):
    ## Get index of best section that best matches query
    if scoring:
        results = scoring.search(query, 1)
        match = data[results[0][0]] if results else "No results"
        print("%-20s %s" % (query, match))

print( ' ' )
print("-" * 70)
print( '(5) Building Semantic index...')

from txtai.embeddings import Embeddings

# Create embeddings model, backed by sentence-transformers & transformers
embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})
embeddings.index(((x, text, None) for x, text in enumerate(data)))
print( '...Index built')
print("-" * 70)
print( ' ')

print( '-' * 70 )
print( '(6) Showing good `semantic` query results on semantic index...' )
print( '-' * 70 )
print( ' ')

print("%-20s %s" % ("Query", "Best Match"))
print("-" * 50)

for query in ("feel good story", 
              "climate change", 
              "public health story",
              "war", 
              "wildlife", 
              "asia", 
              "lucky", 
              "dishonest junk"):
    # Get index of best section that best matches query
    uid = embeddings.search(query, 1)[0][0]

    print("%-20s %s" % (query, data[uid]))  # type: ignore

print( ' ')
print( '-' * 70 )
print( '(7) Surprise! Showing good `keyword` query results on semantic index...' )
print( '-' * 70 )
print( ' ')

for query in ("lottery winner", 
              "canadian iceberg", 
              "number of cases",
              "rising tensions", 
              "park service"):
    # Get index of best section that best matches query
    uid = embeddings.search(query, 1)[0][0]

    print("%-20s %s" % (query, data[uid]))  # type: ignore

print( ' ')
print( '-' * 70 )
print( '-- END --' )
print( '-' * 70 )
print( ' ')
