Below can be found a list of data manulipation scripts that help make this work posible.

# Modules

All scripts have been tested on Python 3.8.2.
The below modules are need to run the scripts.
The scripts were tested on the noted versions, so YMMV.
**Note**: not all modules are required for all scripts.
If this it the first time running the scripts, the modules will need to be installed.
They can be installed by navigating to the `~/code` folder, then using the below code.

* bs4 0.0.1
* lxml 4.5.0
* progressbar2 3.47.0

```{shell}
pip install -r requirments.txt
```

# Scripts

1. [Get bill URLs](./get_bill_urls.py) to convert the list of URLs that were downloaded by hand in JSON format to CSV.
   This will create the clean file `~/data/url_list.csv`.
2. [Get data](./get_data.py) to retrieve both the vote tally and bill text.
   This will create a folder structure under `~/data/votes/{session}-{year}`.
