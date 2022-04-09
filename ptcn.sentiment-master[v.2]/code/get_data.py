import urllib.request
import json
import pathlib
import progressbar as pb
from time import sleep
from bs4 import BeautifulSoup

# string constants for assembling urls
_bill_text_url_format = 'https://www.congress.gov/bill/{session}-congress/{chamber}-bill/{number}/text?format=txt'
_export_segment       = '/export/csv'
_user_agent           = 'VoteAnalysis/1.0'

# pathing
_data_root = pathlib.Path(__file__).parent.joinpath('../data')
_vote_folder = _data_root.joinpath('./votes')
_url_list = _data_root.joinpath('./url_list.csv')
_issues_list = _data_root.joinpath('./issues_list.csv')

# robots.txt delays
_congressgov = 2
_govtrackus = 30

def get_vote_folder(url):
    segments = url.split('/')
    save_as_folder = _vote_folder.joinpath(segments[-2]).joinpath(segments[-1])
    save_as_folder.mkdir(parents = True, exist_ok = True)
    return save_as_folder

def get_vote_status_url(url):
    pass
    return url + _export_segment

def save_vote_status(vote_folder, url):

    save_as_file = vote_folder.joinpath('./votes.csv')

    if pathlib.Path(save_as_file).exists():
        print('Skipping status download...')
        pass
    else:

        content = urlopen_content_safe(url, _govtrackus)

        if content is not None:
            with open(save_as_file, 'wb') as save_as_stream:
                save_as_stream.write(content)
                pass
            pass
        else:
            save_as_file = None
            pass
        pass

    return save_as_file

def get_bill_text_url(url, vote_status_file):

    with open(vote_status_file, 'r', encoding = 'utf-8') as vote_status_stream:
        for line in vote_status_stream:
            line = line.strip().split()
            break
        pass

    chamber = line[0].lower()
    hr = line.index('H.R.')
    number = ''.join(c for c in line[hr+1] if c.isdigit())
    session = str(url.split('/')[-2].split('-')[0]) + 'th'

    return _bill_text_url_format.format(session = session, chamber = chamber, number = number)

def get_bill_text_url_alt(url):
    
    if '/house-bill/' in url:
        alt_url = url.replace('/house-bill/', '/senate-bill/')
        pass
    elif '/senate-bill/' in url:
        alt_url = url.replace('/senate-bill/', '/house-bill/')
        pass
    else:
        alt_url = url
        pass

    return alt_url

def save_bill_text(vote_folder, url):

    save_as_file = vote_folder.joinpath('./bill.txt')

    if pathlib.Path(save_as_file).exists():
        print('Skipping text download...')
        pass
    else:

        content = urlopen_content_safe(url, _congressgov)

        if content is not None:
            content = content.decode('utf-8')
            soup = BeautifulSoup(content, features = 'lxml')
            bill_text = soup.find(id = "billTextContainer")
            if bill_text is None:
                print('bill has no text url: {url}'.format(url = url))
                pass
            else:
                with open(save_as_file, 'w') as save_as_stream:
                    save_as_stream.write(str(bill_text.next))
            pass
        else:
            save_as_file = None
            pass
        pass

    return save_as_file

def urlopen_content_safe(url, delay):

    req = urllib.request.Request(url, headers = {'User-Agent': _user_agent})
    content = None

    try:
        response = urllib.request.urlopen(req)
        response_code = response.getcode()
        if response_code == 200:
            content = response.read()
            pass
        else:
            print('could not open [{code}] url: {url}'.format(code = response_code, url = url))
            pass
    except urllib.error.HTTPError as e:
        print('could not open [{code}] url: {url}'.format(code = e.code, url = url))
        pass
    except urllib.error.URLError as e:
        print('could not open [{reason}] url: {url}'.format(reason = e.reason, url = url))
        pass

    robots_delay(delay)
    return content

def robots_delay(delay):
    widgets = [ 'robots.txt delay ', pb.Percentage(), ' ', pb.Bar(marker = '.', left = '[', right = ']'), ' ', pb.ETA() ]
    with pb.ProgressBar(widgets = widgets, max_value = delay) as bar:
        for x in range(delay):
            bar.update(x)
            sleep(1)
        pass
    pass

with open(_issues_list, 'w') as issues_list:
    with open(_url_list, 'r') as url_list:
        for url in url_list:
            url = url.strip()
            print('processing ' + url)
            vote_folder = get_vote_folder(url)
            vote_status_url = get_vote_status_url(url)
            if vote_status_url is not None:
                vote_status_file = save_vote_status(vote_folder, vote_status_url)
                if vote_status_file is None:
                    issues_list.write(vote_status_url + '\n')
                    pass
                else:
                    bill_text_url = get_bill_text_url(url, vote_status_file)
                    if bill_text_url is not None:
                        bill_text_file = save_bill_text(vote_folder, bill_text_url)
                        if bill_text_file is None:
                            bill_text_url_alt = get_bill_text_url_alt(bill_text_url)
                            bill_text_file = save_bill_text(vote_folder, bill_text_url_alt)
                            if bill_text_file is None:
                                issues_list.write(bill_text_url + '\n')
                                pass
                            pass
                        pass
                    pass
                pass
            pass
        pass
