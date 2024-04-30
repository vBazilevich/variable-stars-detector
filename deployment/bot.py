import os
import sys
import io
import logging
from dotenv import load_dotenv
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
import httpx


load_dotenv()
TOKEN = os.getenv('TG_TOKEN')

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message) -> None:
    await message.answer(f'Welcome to VariableStarsDetectorBot!')

@dp.message(F.document & (F.document.mime_type == 'text/plain'))
async def files_handler(message, bot: Bot):
    file = io.BytesIO()
    await message.answer('Downloading the file...')
    await bot.download(message.document.file_id, destination=file)

    await message.answer('File is received... Starting processing')

    # Send request to RF
    async with httpx.AsyncClient() as client:
        files = {'file': file}
        rf_request = client.post('http://localhost:14001/predict', files=files)

        await message.answer(f'Random forest: {(await rf_request).text}')

@dp.message()
async def default_handler(message):
    logging.info(message.document)
    await message.answer('Unsupported message format...')

async def main():
    bot = Bot(token=TOKEN)
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
