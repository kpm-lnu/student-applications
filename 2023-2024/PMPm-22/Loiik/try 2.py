import json
import time
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service


def login(driver, email, password):
    email_input = driver.find_element(By.NAME, "email")
    email_input.clear()
    email_input.send_keys(email)

    password_input = driver.find_element(By.NAME, "pass")
    password_input.clear()
    password_input.send_keys(password)

    login_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[1]/div[1]/div/div/div/div[2]/div/div[1]/form/div[2]/button")
    print(login_button)
    login_button.click()


def get_number_of_posts(driver, link, hashtag):
    driver.get(link)
    time.sleep(2)

    login(driver, "", "")

    time.sleep(10)

    driver.get("https://www.facebook.com/hashtag/" + hashtag + "/")
    # search_label = driver.find_element(By.XPATH, "/html/body/div[1]/div/div[1]/div/div[2]/div[3]/div/div/div/div/div/label")
    # search_label.click()
    #
    # search_input = driver.find_element(By.XPATH, "/html/body/div[1]/div/div[1]/div/div[2]/div[3]/div/div/div[1]/div/div/label/input")
    # search_input.send_keys('#' + hashtag)
    # search_input.send_keys(Keys.RETURN)

    time.sleep(2)

    posts = driver.find_elements(By.XPATH, "//div[contains(@class, 'post')]")
    num_posts = len(posts)

    return num_posts


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    social_network_links = [
        "https://www.facebook.com/"
    ]
    hashtag = "love"
    start_date = datetime(2024, 5, 1)
    end_date = datetime(2024, 5, 7)
    data = {}

    service = Service('./chromedriver.exe')
    driver = webdriver.Chrome(service=service)

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        data[date_str] = {}

        for link in social_network_links:
            num_posts = get_number_of_posts(driver, link, hashtag)
            data[date_str][link] = num_posts

        current_date += timedelta(days=1)

    filename = f"{hashtag}.json"
    save_to_json(data, filename)

    driver.quit()

if __name__ == "__main__":
    main()
