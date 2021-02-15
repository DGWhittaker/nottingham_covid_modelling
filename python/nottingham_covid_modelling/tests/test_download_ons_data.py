import os
import sys
from unittest import mock

import pytest
import requests_mock
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.download_ons_data import download_and_process_ons


@pytest.fixture(scope='session')
def reference_folder():
    """ Returns the path were the reference files for tests are stored in an OS independent way."""
    return os.path.join(MODULE_DIR, 'tests', 'reference_files')


def test_help(capsys, reference_folder):
    """ Check script usage"""
    testargs = ["download_ons_data", "-h"]
    with mock.patch.object(sys, 'argv', testargs):
        try:
            download_and_process_ons()
        except SystemExit:
            pass  # We expect this to print usage and exit
    captured = capsys.readouterr()
    output = str(captured.out)
    print(output)
    expected = open(os.path.join(reference_folder, 'download_ons_data_help.txt'), 'r').read()
    assert output == expected


def test_wrong_download_location():
    """ Checks we get an error if we try to download a non-existing location """
    testargs = ["download_ons_data", "-download-location", "http://blabla.testcovid.com"]
    with mock.patch.object(sys, 'argv', testargs):
        with pytest.raises(IOError, match='Could not download ONS spreadsheet'):
            download_and_process_ons()


def test_wrong_download_location2():
    """ Checks we get an error if we try to download an existing location which doesn't have the spreadsheet"""
    testargs = ["download_ons_data", "-download-location", "http://www.google.com"]
    with mock.patch.object(sys, 'argv', testargs):
        with pytest.raises(IOError, match='Could not download ONS spreadsheet'):
            download_and_process_ons()


def test_xlsx_without_tab(reference_folder):
    """ Check we get an error if the worksheet doesn't exist"""
    mock_xlsx = open(os.path.join(reference_folder, 'missing_tab.xlsx'), 'rb').read()
    testargs = ["download_ons_data", "-download-location", "http://blabla.testcovid.com"]
    with mock.patch.object(sys, 'argv', testargs):
        with requests_mock.Mocker() as m:
            m.get("http://blabla.testcovid.com", content=mock_xlsx)
            with pytest.raises(IOError, match='Cannot find worksheet'):
                download_and_process_ons()


def test_xlsx_missing_date_column(reference_folder):
    """ Check we get an error if the date columns doesn't exist"""
    mock_xlsx = open(os.path.join(reference_folder, 'missing_date.xlsx'), 'rb').read()
    testargs = ["download_ons_data", "-download-location", "http://blabla.testcovid.com"]
    with mock.patch.object(sys, 'argv', testargs):
        with requests_mock.Mocker() as m:
            m.get("http://blabla.testcovid.com", content=mock_xlsx)
            with pytest.raises(KeyError, match='date'):
                download_and_process_ons()


def test_xlsx_missing_date_data(reference_folder):
    """ Check we get an error if the date columns doesn't exist"""
    mock_xlsx = open(os.path.join(reference_folder, 'missing_date_data.xlsx'), 'rb').read()
    testargs = ["download_ons_data", "-download-location", "http://blabla.testcovid.com"]
    with mock.patch.object(sys, 'argv', testargs):
        with requests_mock.Mocker() as m:
            m.get("http://blabla.testcovid.com", content=mock_xlsx)
            with pytest.raises(AssertionError,
                               match='Expecting the downloaded to contain data but got none!'):
                download_and_process_ons()


def test_xlsx_missing_deaths_column(reference_folder):
    """ Check we get an error if the date data doesn't exist"""
    mock_xlsx = open(os.path.join(reference_folder, 'missing_deaths_column.xlsx'), 'rb').read()
    testargs = ["download_ons_data", "-download-location", "http://blabla.testcovid.com"]
    with mock.patch.object(sys, 'argv', testargs):
        with requests_mock.Mocker() as m:
            m.get("http://blabla.testcovid.com", content=mock_xlsx)
            with pytest.raises(KeyError, match='dailys_deaths'):
                download_and_process_ons()


def test_xlsx_missing_deaths_data(reference_folder):
    """ Check we get an error if the deaths data doesn't exist"""
    mock_xlsx = open(os.path.join(reference_folder, 'missing_deaths_column_data.xlsx'), 'rb').read()
    testargs = ["download_ons_data", "-download-location", "http://blabla.testcovid.com"]
    with mock.patch.object(sys, 'argv', testargs):
        with requests_mock.Mocker() as m:
            m.get("http://blabla.testcovid.com", content=mock_xlsx)
            with pytest.raises(TypeError, match='unsupported operand type(.)*'):
                download_and_process_ons()


def test_xlsx_deaths_going_down(reference_folder):
    """ Check we get an error if the deaths are going down compared to previous file"""
    mock_xlsx = open(os.path.join(reference_folder, 'publishedweek182020.xlsx'), 'rb').read()
    testargs = ["download_ons_data", "-download-location", "http://blabla.testcovid.com",
                "--output", os.path.join(reference_folder, 'publishedweek182020_numbers_raised.csv')]
    with mock.patch.object(sys, 'argv', testargs):
        with requests_mock.Mocker() as m:
            m.get("http://blabla.testcovid.com", content=mock_xlsx)
            with pytest.raises(AssertionError,
                               match='We do not expect deaths to go down compared to the previous data'):
                download_and_process_ons()


def test_xlsx_deaths_going_up(reference_folder):
    """ Check we get an error if the deaths are going up too much compared to previous file"""
    mock_xlsx = open(os.path.join(reference_folder, 'publishedweek182020.xlsx'), 'rb').read()
    testargs = ["download_ons_data", "-download-location", "http://blabla.testcovid.com",
                "--output", os.path.join(reference_folder, 'publishedweek182020_numbers_lowered.csv')]
    with mock.patch.object(sys, 'argv', testargs):
        with requests_mock.Mocker() as m:
            m.get("http://blabla.testcovid.com", content=mock_xlsx)
            with pytest.raises(AssertionError,
                               match="We do not expect more than a 5% in deaths over previously."):
                download_and_process_ons()


def test_without_csv(reference_folder):
    """ Check we get an error if the the csv file doesn't exist (with sanity check on)"""
    mock_xlsx = open(os.path.join(reference_folder, 'publishedweek182020.xlsx'), 'rb').read()
    testargs = ["download_ons_data", "-download-location", "http://blabla.testcovid.com", "--output", 'bla.csv']
    with mock.patch.object(sys, 'argv', testargs):
        with requests_mock.Mocker() as m:
            m.get("http://blabla.testcovid.com", content=mock_xlsx)
            with pytest.raises(AssertionError,
                               match='provided file should exist so we can sanity check the newly downloaded data. '
                                     'Run script with --skip_sanity_check to skip this test.'):
                download_and_process_ons()


def test_against_reference(tmp_path, reference_folder):
    """ Test script gives expected CSV for given reference spreadsheet """
    mock_xlsx = open(os.path.join(reference_folder, 'publishedweek182020.xlsx'), 'rb').read()
    downloaded_csv = os.path.join(tmp_path, "downloaded.csv")
    testargs = ["download_ons_data", "-download-location", "http://blabla.testcovid.com", "--output",
                downloaded_csv, "--skip_sanity_check"]
    with mock.patch.object(sys, 'argv', testargs):
        with requests_mock.Mocker() as m:
            m.get("http://blabla.testcovid.com", content=mock_xlsx)
            download_and_process_ons()

    expected_csv = open(os.path.join(reference_folder, 'publishedweek182020.csv'), 'r').read()
    downloaded_csv = open(downloaded_csv, 'r').read()
    assert downloaded_csv == expected_csv


def test_generate_new_daily_deaths(tmp_path):
    """ Check we can generate new csv without error (including sanity check in the script)
    """
    downloaded_csv = os.path.join(tmp_path, "downloaded.csv")
    testargs = ["download_ons_data", "--output", downloaded_csv, "--skip_sanity_check"]
    with mock.patch.object(sys, 'argv', testargs):
        download_and_process_ons()
