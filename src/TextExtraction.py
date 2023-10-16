from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime

url = "https://www.centcom.mil/MEDIA/NEWS-ARTICLES/News-Article-View/Article/3376118/merchant-vessel-seized-by-irgcn-in-gulf-of-oman/"
html = urlopen(url).read()
soup = BeautifulSoup(html, features="html.parser")

# kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # rip it out

# get text
text = soup.get_text()

# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)

filename = datetime.datetime.now()

with open("DataExtraction/centcom/" + str(filename) + ".txt", 'w', encoding='utf-8') as f:
    f.write(text)
    f.close