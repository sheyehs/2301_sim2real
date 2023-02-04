import gdown

# same as the above, and you can copy-and-paste a URL from Google Drive with fuzzy=True
url = "https://drive.google.com/file/d/14am9czKyOMQ9xZ-BZ-3gIQbmyVfunFOu/view?usp=share_link"
output = "2022-11-15.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)