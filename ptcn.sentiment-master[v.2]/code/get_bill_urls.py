import json
import pathlib
from bs4 import BeautifulSoup

url_format = 'https://www.govtrack.us{segment}\n'

data_root = pathlib.Path(__file__).parent.joinpath('../data')
list_of_bills_path = data_root.joinpath('./list_of_bills')
url_list = data_root.joinpath('./url_list.csv')

def get_list_of_bills(json_data):
    for result in json_data['results']:
        soup = BeautifulSoup(result, features = 'lxml')
        for anchor in soup.find_all('a'):
            if ('href' in anchor.attrs) and (anchor.next is not None) and (anchor.next.startswith('H.R. ')):
                yield anchor.attrs['href']
                break
            pass
        pass

with open(url_list, 'w', encoding = 'utf-8') as file_out:
    for entry in list_of_bills_path.iterdir():
        if entry.is_file() and entry.suffix == '.json':
            print('processing ' + entry.stem)
            with open(entry, 'r') as json_file:
                contents = json_file.read()
                contents = json.loads(contents)
                urls = get_list_of_bills(contents)
                file_out.writelines([url_format.format(segment = url) for url in urls])
                pass
            pass
        pass
    pass