import argparse
import datetime
import os
import zipfile
from io import BytesIO

import openpyxl
import requests
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.lib._csv_utils import read_csv, write_to_csv


ONS_LOCATION = 'https://www.ons.gov.uk/file?uri=%2fpeoplepopulationandcommunity%2fbirthsdeathsandmarriages%2fdeaths%2'\
    'fdatasets%2fweeklyprovisionalfiguresondeathsregisteredinenglandandwales%2f{year}/publishedweek{week}{year}.xlsx'
MAX_INCREASE_PERCENTAGE = 5


def download_and_process_ons():
    parser = argparse.ArgumentParser()

    parser.add_argument('-download-location', dest='download_location', type=str, default=ONS_LOCATION,
                        help='Optional: the url for downloading the ons spreadsheet (defaults to the ons website).')
    parser.add_argument('--output', dest='csv_file', type=str,
                        default=os.path.join(MODULE_DIR, '..', '..', 'data', 'archive', 'current', 'ONS_daily_deaths.csv'),
                        help='Optional: full path of the file to save the resulting csv to.')
    parser.add_argument('--skip_sanity_check', dest='sanity_check', action='store_false', default=True,
                        help='The script sanity checks the generated CVS and compares it against the existing one in'
                             ' the location provided, this option allows this check to be skipped.')
    parser.add_argument('--max_increase_percentage_existing', dest='max_increase_percentage',
                        default=MAX_INCREASE_PERCENTAGE,
                        help='Percentage the sanity check allows deaths to increase in downloaded data compared to '
                             'existing data (default: ' + str(MAX_INCREASE_PERCENTAGE) + '%%).')
    args = parser.parse_args()

    print('csv file: ' + str(args.csv_file))
    # Download ONS spreadsheet
    # Starting from today work backwards to previous weeks until we find one
    day = datetime.datetime.today()
    response = None
    for i in range(10):  # probably don't need to go back more
        year = day.strftime('%Y')
        week = day.strftime('%V')
        url = args.download_location.format(year=year, week=week)  # plus in year and week in doanlod_url
        try:
            response = requests.get(url, allow_redirects=True, timeout=1.0)
            if response.ok and response.text != '':
                wb_obj = openpyxl.load_workbook(filename=BytesIO(response.content))
                print('Latest ONS spreadsheet is from: ' + str(year) + ' week: ' + str(week))
                break  # we've found our spreadsheet
        except (zipfile.BadZipFile, requests.exceptions.ConnectionError):
            pass  # Failed to download but keep trying year/week combos
        if response is not None:
            response.close()  # this wasn't it, close this connection ready for a retry
            response = None
        day = day - datetime.timedelta(days=7)
    else:
        raise IOError('Could not download ONS spreadsheet')

    new = False
    try:
        sheet = wb_obj['Covid-19 - E&W comparisons']
    except KeyError:
        try:
            sheet = wb_obj['Covid-19 - Daily occurrences']
            new = True
        except KeyError:
            print("can't find! worksheet")
            raise IOError('Cannot find worksheet')
            
    # find relevant colum header information
    rows = list(sheet.rows)
    header_row_index = -1  # start of the table
    column_headers = {}  # storingcolumn headers and their indices for easy access
    if new:
        columns_to_exclude = ('uk', 'scotland', 'northern ireland', 'north east', 'north west', 'yorkshire and the humber', 'east midlands', 'west midlands', 'east', 'london', 'south east', 'south west')
        text_required_in_column = tuple('england and wales')
    else:
        columns_to_exclude = ('gov.uk', 'registration', 'none', '(england only)', '(wales only)')
        text_required_in_column = ('deaths', 'actual date of death')

    #  goes through the rows checking all columns until it finds 'date' and then looks for the deaths number
    # e.g. ONS deaths by actual date of death – registered by 23 May
    # as we can't predict the date we instead exclude stuff we definately don't want and pick the last of the remaining.
    # exclude columns if text includes gov.uk, registration (we want date occurred), (England only) and (Wales only)

    for row_index, row in enumerate(rows):
        column_headers = {}
        for column_index, _ in enumerate(list(sheet.columns)):
            cell_val = str(row[column_index].value).lower().strip()
            if cell_val == 'date':
                header_row_index = row_index
                column_headers['date'] = column_index
            elif not any(excl in cell_val for excl in columns_to_exclude) \
                    and all(excl in cell_val for excl in text_required_in_column):
                column_headers['dailys_deaths'] = column_index
        if header_row_index != -1:
            break

    # now we know from what row to start and in which column the date and deaths numbers are
    # go through and pull out death numbers
    # deaths are cumulative so today = cummulative_today - cumulative_yesterday
    death_data = []
    for row_index in range(header_row_index + 1, len(rows)):
        date = rows[row_index][column_headers['date']].value
        try:
            date  = datetime.datetime.strptime(str(date), "%d/%m/%Y")
        except ValueError:
            pass  # date was already in date format

        if isinstance(date, datetime.datetime):  # skip rows where there isn't a date
            dailys_deaths = rows[row_index][column_headers['dailys_deaths']].value
            date = date.strftime('%Y-%m-%d')  # format date for printing
            if row_index - 1 > header_row_index and not new:  # assume dates are in order and subtract cumulative from yesterday
                death_data.append(
                    {'date': date,
                     'dailys_deaths': dailys_deaths - rows[row_index - 1][column_headers['dailys_deaths']].value})
            else:
                death_data.append({'date': date, 'dailys_deaths': dailys_deaths})

    # Sanity check downloaded data against existing data
    if args.sanity_check:
        assert len(death_data) > 1, "Expecting the downloaded to contain data but got none!"
        assert os.path.isfile(args.csv_file), 'provided file should exist so we can sanity check the newly downloaded '\
            'data. Run script with --skip_sanity_check to skip this test.'  # check file exists
        # check against existing before writing
        existing_data = read_csv(args.csv_file)
        assert len(existing_data) > 1, 'Expecting the existing csv file to contain data'
        for row in existing_data[1:]:  # loop though existing data skipping header line
            date = row[0]
            # find same date in generated file
            for gen_index, gen_row in enumerate(death_data):
                if gen_row['date'] == row[0]:
                    assert (gen_row['dailys_deaths'] + 5) >= int(row[1]), \
                        'We do not expect deaths to go down compared to the previous data.'
                    assert gen_row['dailys_deaths'] * (args.max_increase_percentage / 100) <= int(row[1]), 'We do not'\
                        ' expect more than a ' + str(args.max_increase_percentage) + '% in deaths over previously.'

    csv_file = os.path.abspath(args.csv_file)  # get absolute path
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)  # make sure the path exists
    write_to_csv(death_data, csv_file, ['date', 'dailys_deaths'])
