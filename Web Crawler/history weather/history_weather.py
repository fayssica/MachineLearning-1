import requests
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame
import numpy as np


class History_weather():
    def begin_url(self, url):
        all_weather_index = self.get_all_weather_index(url)
        all_weather_index = all_weather_index[:30]
        print("Get all weather index!")
        result_weather = self.get_all_weather(all_weather_index)
        result_weather.to_csv('all_weather.csv', index=False)
        print('Save all weather success!')

    def get_soup(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/602.4.8 \
            (KHTML, like Gecko) Version/10.0.3 Safari/602.4.8'}
        html = requests.get(url, headers)
        soup = BeautifulSoup(html.text, 'lxml')
        return soup

    def get_all_weather_index(self, url):
        all_weather_soup = self.get_soup(url)
        all_weather_index = all_weather_soup.find(
            'div', class_='tqtongji1').find_all('a')
        return all_weather_index

    def get_all_weather(self, all_weather_index):
        all_month_weather = list()
        day_weather = list()
        for weather in all_weather_index:
            month_url = weather['href']
            month_name = weather.get_text()
            month_weather_soup = self.get_soup(month_url)
            month_weather = month_weather_soup.find(
                'div', class_='tqtongji2').find_all('li')
            day_weather_url = month_weather_soup.find(
                'div', class_='tqtongji2').find_all('a')
            for day in day_weather_url:
                day_url = day['href']
                day_soup = self.get_soup(day_url)
                day_text = day_soup.find(
                    'div', class_='history_sh').find_all('span')
                for i in day_text:
                    day_weather.append(i.get_text())
            weather_list = list()
            for i in month_weather:
                weather_list.append(i.get_text())
            weather_list = weather_list[6:]
            all_month_weather = all_month_weather + weather_list
            print("Get weather :" + month_name)
        day_weather = np.array(day_weather).reshape(-1, 8)
        day_weather = DataFrame(day_weather, columns=[
            'Ultraviolet', 'Dress', 'Travel', 'Comfort_level',
            'Morning_exercise', 'Car_wash', 'Drying_index', 'Breath_allergy'])
        all_month_weather = np.array(all_month_weather).reshape(-1, 6)
        all_month_weather = DataFrame(all_month_weather, columns=[
                                      'Date', 'Max_temp', 'Min_temp', 'Whether', 'Wind_direction', 'Wind_power'])
        print(all_month_weather.shape)
        print(day_weather.shape)
        result_weather = pd.merge(
            all_month_weather, day_weather, left_index=True, right_index=True)
        return result_weather

history_weather = History_weather()
begin_url = 'http://lishi.tianqi.com/yangzhong/index.html'
history_weather.begin_url(begin_url)
