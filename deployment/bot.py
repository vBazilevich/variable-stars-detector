import os
import sys
import io
import logging
from dotenv import load_dotenv
import asyncio
import aiogram
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
import httpx
import prettytable as pt
import json


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
        resnet_request = client.post('http://localhost:14002/predict', files=files)

        # TODO: check response code before parsing
        rf_request = json.loads((await rf_request).text)
        resnet_request = json.loads((await resnet_request).text)

        table = pt.PrettyTable(['Class', 'Random Forest', 'ResNet'], float_format='.3')
        for var_class in rf_request.keys():
            table.add_row([var_class, rf_request[var_class], resnet_request[var_class]])

        await message.answer(f'```{table}```', parse_mode='MarkdownV2')

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
