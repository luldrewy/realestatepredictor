import requests 
import json
from bs4 import BeautifulSoup
import pandas as pd



class Scraper:
    
    
    def getData(self, searchQuery):
        headers = {
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}
        params = {"search": str(searchQuery)}

        search_url = 'https://www.zillow.com/homes/'
        url = search_url + params["search"] 

        r = requests.get(url, headers= headers)
        soup = BeautifulSoup(r.content, "html.parser")
        
        print(soup.body)
            

        data = json.loads(
            soup.select_one("script[data-zrr-shared-data-key]")
            .contents[0].strip("!<>-")
        )
        
        all_data = data['cat1']['searchResults']['listResults']


        priceList = []
        addressList = []
        link_to_buyList = []
        sizeList = []
        bedsList = []
        bathsList = []
        zipCodeList = []
        lotSizeList = []

        for i in range(len(all_data)):
            #some items have the 'price' key nested inside units key, while others have simply inside data key
            try:
                price = all_data[i]['units'][0]['price']
                size = all_data[i]['area']
                beds = all_data[i]['beds']
                baths = all_data[i]['baths']
                lotSize = all_data[i]["hdpData"]["homeInfo"]["lotAreaValue"]
                zipCode = int(all_data[i]["hdpData"]["homeInfo"]["zipcode"])

            except KeyError:
                price = all_data[i]['price']
                size = all_data[i]['area']
                beds = all_data[i]['beds']
                baths = all_data[i]['baths']
                try:
                    lotSize = all_data[i]["hdpData"]["homeInfo"]["lotAreaValue"]
                except KeyError:
                    lotSize = float("NaN")
                zipCode = int(all_data[i]["hdpData"]["homeInfo"]["zipcode"])
            address = all_data[i]['address']
            
            specialChar = ["$", ",","M","K", "+","."]
            price = "".join(filter(lambda char: char not in specialChar, price))
            price = int(price)
            

                

            link = all_data[i]['detailUrl']
            if 'http' not in link:
                link_to_buy = f"https://www.zillow.com{link}"
            else:
                link_to_buy = link
            
            priceList.append(price)
            addressList.append(address)
            link_to_buyList.append(link_to_buy)
            sizeList.append(size)
            bathsList.append(baths)
            bedsList.append(beds)
            zipCodeList.append(zipCode)
            lotSizeList.append(lotSize)
            
        
        df = pd.DataFrame(zip(bedsList, bathsList, sizeList,zipCodeList, priceList, addressList, link_to_buyList) , columns= ["beds","baths", "size","zip_code","price", "address","link"])
        pd.set_option("display.width", 1000)
        df.to_csv(r'/Users/andrew/Desktop/LearningC++/LearningPython/MachineLearning/ZillowData.csv', index= False)                         
        
        return df
    
    

        

