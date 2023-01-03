import bs4
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os   
import time
""" options = webdriver.ChromeOptions()
options.add_experimental_option('detach', True)
driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install())) 
"""
options = webdriver.ChromeOptions()
options.add_experimental_option('detach', True)

#driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
def downloadImage(url, folderName, num):
    response = requests.get(url)
    if response.status_code == 200:   
        with open(os.path.join(folderName, str(num)+".jpg"), 'wb') as file:
            file.write(response.content)

driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))
searchURL = 'https://www.google.com/search?q=rice+leaf&source=Inms&tbm=isch'
driver.get(searchURL)
folderPath = "ScrapImage/bg"
a = input('Waiting for user input to start....')
driver.execute_script("window.scrollTo(0, 0);") 

page_html = driver.page_source
pageSoup = bs4.BeautifulSoup(page_html, 'html.parser')
containers = pageSoup.findAll('div',{'class':"isv-r PNCib MSM1fd BUooTd"})
len_containers = len(containers)
print(len_containers)

for i in range(1, len_containers+1):
    if i % 25 == 0:
        continue
    xpath = '''//*[@id="islrg"]/div[1]/div[%s]'''%(i)
    previewImageXPath = """//*[@id="islrg"]/div[1]/div[%s]/a[1]/div[1]/img"""%(i)
    previewImageElement = driver.find_element('xpath', previewImageXPath)
    previewImageURL = previewImageElement.get_attribute('src')
    driver.find_element('xpath', xpath).click()
    
    timeStarted = time.time()
    while True:
        
        imageElement = driver.find_element('xpath',"""//*[@id="Sva75c"]/div/div/div[2]/div[2]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img""")
        imageURL = imageElement.get_attribute('src')
        if imageURL != previewImageURL:
            break
        else: 
            currentTime = time.time()
            if currentTime - timeStarted > 10:
                print("Timeout! Will download a lower resolution image and move onto the next one")
                break
    try: 
        downloadImage(imageURL,folderPath,i)
        print("Downloaded element %s out of %s total. URL: %s" % (i, len_containers + 1, imageURL))
    except:
        print("Couldn't download an image %s, continuing downloading the next one"%(i))
    
