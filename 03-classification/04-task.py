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

# print(spam_emails[0].get_content().strip())
spam_filenames[0]


# %%
spam_emails[0].get_content().strip()