# Fake_News_Detection

## Onderzoeksrapport
Een onderzoek over de relatie tussen metadata en nepnieuws

[Onderzoeksrapport](https://docs.google.com/document/d/1p_0LL0oiC-zCGx--G1a6unonNnChmGRcjt1PUfZCMbg/edit#)

## Project local werkend krijgen

**prerequisites**
- Installeer Python 3.10.5
- installeer [Graphviz](https://graphviz.gitlab.io/download/)
- Om gebruik te maken van het BERT-model in o.a. concatenate.ipynb, run bert.ipynb om de weights te krijgen <strong>of</strong> download bert-text-metadata-weights.h5 en plaats het in het pad 'models/text'. Voor het nn.ipynb moet hetzelfde gebeuren, alleen moet het h5 bestand geplaatst worden in het mapje 'models/metadata'. 
- Om het project te runnen moeten ten eerste alle libraries geinstalleerd worden. Om alle libraries te verkrijgen, is de requirements.txt file beschikbaar. Maak een nieuwe environment aan en run de volgende command: pip install -r requirements.txt. 
- Zorg ervoor dat de datasets van de vorige teams beschikbaar zijn. Vraag aan de opdrachter voor deze bestanden. Plaats deze in een map genaamd 'previous_datasets'. Hieronder is de file structure te zien:

```
├── data
│   ├── analysis
│   ├── functions
│   ├── previous_datasets
|   |   ├── CMU-Miscov19
|   |   |   ├── modified
|   |   |   |   ├── MisCov_Complete.csv
|   |   ├── combined
|   |   |   ├── tweets_fng.csv
|   |   ├── CONSTRAINT_2021
|   |   |   ├── original
|   |   |   |   ├── Constraint_English_Test.csv
|   |   |   |   ├── Constraint_English_Train.csv
|   |   |   |   ├── Constraint_English_Val.csv
│   ├── visuals
```

<span style="color:red">!note:</span> de csv bestanden die nodig zijn heten:
- MisCov_Complete.csv
- tweets_fng.csv
- Constraint_English_Test.csv
- Constraint_English_Train.csv
- Constraint_English_Val.csv

**Project runnen**
> Run 'preprocessing_2022.ipynb'
>> Nu alle libraries local in je environment staan en de filestructure goed staat, kan de 'preprocessing_2022.ipynb' gerunned worden. Na het runnen van de file wordt er een map aangemaakt genaamd 'cleaned_dataset'. Hierin worden verschillende csv bestanden gecreëerd.
 
> Run 'feature_selection.ipynb'
>> Ook dit creëert een nieuwe bestand genaamd: 'selected_data'. Deze files worden ook gebruikt door verschillende classifiers.

<br>

**congratulations !**

Nu zijn alle datasets beschikbaar om de modellen te runnen. 






