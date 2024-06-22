from aiogram import Bot, Dispatcher, types
import os
import asyncio 
import logging
from inference import generate_sample


bot = Bot(os.environ["TOKEN"])
dp = Dispatcher()

@dp.message()
async def echo_message(message: types.Message):
    if "клон" in message.text.lower():  # Проверяем наличие слова "клон" в тексте сообщения
        prompt = message.text  # Получаем текст сообщения
        response = generate_sample(prompt)  # Генерируем ответ с помощью нейросети
        await message.answer(text=response)  # Отправляем ответ
async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())