from skbio import DistanceMatrix
import pandas as pd
from skbio.stats.ordination import PCoA
from pylab import figure, axes, pie, title, show
import matplotlib.image as mpimg




dm = DistanceMatrix([[0., 0.21712454, 0.5007512, 0.91769271],
                     [0.21712454, 0., 0.45995501, 0.80332382],
                     [0.5007512, 0.45995501, 0., 0.65463348],
                     [0.91769271, 0.80332382, 0.65463348, 0.]],
                    ['A', 'B', 'C', 'D'])

metadata = {
    'A': {'body_site': 'skin'},
    'B': {'body_site': 'gut'},
    'C': {'body_site': 'gut'},
    'D': {'body_site': 'skin'}}
df = pd.DataFrame.from_dict(metadata, orient='index')

pcoa_results = PCoA(dm).scores()

fig = pcoa_results.plot(df=df, column='body_site',
                        title='Sites colored by body site',
                        cmap='Set1', s=50)

fig.savefig('foo.png')
