# %%
import os
import urllib
import tarfile

DOWNLOAD_ROOT = "https://spamassassin.apache.org/old/publiccorpus"
SPAM_URL = DOWNLOAD_ROOT + "/20021010_spam.tar.bz2"
HAM_URL = DOWNLOAD_ROOT + "/20030228_easy_ham.tar.bz2"
DATA_PATH = os.path.join("datasets", "spam")

def fetch_spam_data(spam_path=DATA_PATH):
    os.makedirs(spam_path, exist_ok=True)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(DATA_PATH, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tgz_file = tarfile.open(path)
        tgz_file.extractall(path=DATA_PATH)
        tgz_file.close()

fetch_spam_data()

# %%
SPAM_DIR = os.path.join(DATA_PATH, "spam")
HAM_DIR = os.path.join(DATA_PATH, "easy_ham")
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20 and not name.startswith('0000.')]
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]

# %%
len(spam_filenames)

# %%
len(ham_filenames)

# %%
import email
import email.policy

def load_email(is_spam, filename):
    directory = SPAM_DIR if is_spam else HAM_DIR
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

spam_emails = [load_email(True, name) for name in spam_filenames]
ham_emails = [load_email(False, name) for name in ham_filenames]


# %%
ham_emails[1].get_content().strip()

# %%
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

# %%
from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

# %%
structures_counter(ham_emails).most_common()

# %%
structures_counter(spam_emails).most_common()

# %%
for header, value in spam_emails[0].items():
    print(header, ",", value)

# %%
spam_emails[0]["Subject"]

# %%
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(ham_emails + spam_emails)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
import re
from html import unescape

def html_to_plan_text(html):
    text = re.sub('<head.*>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

# %%
html_spam_emails = [email for email in X_train[y_train == 1]
                    if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")

# %%
print(html_to_plan_text(sample_html_spam.get_content())[:1000], "...")

# %%
def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plan_text(html)

    