# ML-algorithms
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP9a2ecO2lw558fwxwzi00U",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/pujashah2170/Grp-14-Assignment-1/blob/main/Breast_cancer_detection_using_various_ML_algorithms.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "idf8GDZRFvXf"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Loading dataset"
      ],
      "metadata": {
        "id": "FStO4Ff_GEL0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "upload=files.upload()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 72
        },
        "id": "XkrQ5y0oGJXe",
        "outputId": "90604e52-2cc3-4d86-ec42-ba168cda0ad7"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-76e605c9-e744-438b-92d4-681988f3db15\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-76e605c9-e744-438b-92d4-681988f3db15\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script>// Copyright 2017 Google LLC\n",
              "//\n",
              "// Licensed under the Apache License, Version 2.0 (the \"License\");\n",
              "// you may not use this file except in compliance with the License.\n",
              "// You may obtain a copy of the License at\n",
              "//\n",
              "//      http://www.apache.org/licenses/LICENSE-2.0\n",
              "//\n",
              "// Unless required by applicable law or agreed to in writing, software\n",
              "// distributed under the License is distributed on an \"AS IS\" BASIS,\n",
              "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
              "// See the License for the specific language governing permissions and\n",
              "// limitations under the License.\n",
              "\n",
              "/**\n",
              " * @fileoverview Helpers for google.colab Python module.\n",
              " */\n",
              "(function(scope) {\n",
              "function span(text, styleAttributes = {}) {\n",
              "  const element = document.createElement('span');\n",
              "  element.textContent = text;\n",
              "  for (const key of Object.keys(styleAttributes)) {\n",
              "    element.style[key] = styleAttributes[key];\n",
              "  }\n",
              "  return element;\n",
              "}\n",
              "\n",
              "// Max number of bytes which will be uploaded at a time.\n",
              "const MAX_PAYLOAD_SIZE = 100 * 1024;\n",
              "\n",
              "function _uploadFiles(inputId, outputId) {\n",
              "  const steps = uploadFilesStep(inputId, outputId);\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  // Cache steps on the outputElement to make it available for the next call\n",
              "  // to uploadFilesContinue from Python.\n",
              "  outputElement.steps = steps;\n",
              "\n",
              "  return _uploadFilesContinue(outputId);\n",
              "}\n",
              "\n",
              "// This is roughly an async generator (not supported in the browser yet),\n",
              "// where there are multiple asynchronous steps and the Python side is going\n",
              "// to poll for completion of each step.\n",
              "// This uses a Promise to block the python side on completion of each step,\n",
              "// then passes the result of the previous step as the input to the next step.\n",
              "function _uploadFilesContinue(outputId) {\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  const steps = outputElement.steps;\n",
              "\n",
              "  const next = steps.next(outputElement.lastPromiseValue);\n",
              "  return Promise.resolve(next.value.promise).then((value) => {\n",
              "    // Cache the last promise value to make it available to the next\n",
              "    // step of the generator.\n",
              "    outputElement.lastPromiseValue = value;\n",
              "    return next.value.response;\n",
              "  });\n",
              "}\n",
              "\n",
              "/**\n",
              " * Generator function which is called between each async step of the upload\n",
              " * process.\n",
              " * @param {string} inputId Element ID of the input file picker element.\n",
              " * @param {string} outputId Element ID of the output display.\n",
              " * @return {!Iterable<!Object>} Iterable of next steps.\n",
              " */\n",
              "function* uploadFilesStep(inputId, outputId) {\n",
              "  const inputElement = document.getElementById(inputId);\n",
              "  inputElement.disabled = false;\n",
              "\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  outputElement.innerHTML = '';\n",
              "\n",
              "  const pickedPromise = new Promise((resolve) => {\n",
              "    inputElement.addEventListener('change', (e) => {\n",
              "      resolve(e.target.files);\n",
              "    });\n",
              "  });\n",
              "\n",
              "  const cancel = document.createElement('button');\n",
              "  inputElement.parentElement.appendChild(cancel);\n",
              "  cancel.textContent = 'Cancel upload';\n",
              "  const cancelPromise = new Promise((resolve) => {\n",
              "    cancel.onclick = () => {\n",
              "      resolve(null);\n",
              "    };\n",
              "  });\n",
              "\n",
              "  // Wait for the user to pick the files.\n",
              "  const files = yield {\n",
              "    promise: Promise.race([pickedPromise, cancelPromise]),\n",
              "    response: {\n",
              "      action: 'starting',\n",
              "    }\n",
              "  };\n",
              "\n",
              "  cancel.remove();\n",
              "\n",
              "  // Disable the input element since further picks are not allowed.\n",
              "  inputElement.disabled = true;\n",
              "\n",
              "  if (!files) {\n",
              "    return {\n",
              "      response: {\n",
              "        action: 'complete',\n",
              "      }\n",
              "    };\n",
              "  }\n",
              "\n",
              "  for (const file of files) {\n",
              "    const li = document.createElement('li');\n",
              "    li.append(span(file.name, {fontWeight: 'bold'}));\n",
              "    li.append(span(\n",
              "        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +\n",
              "        `last modified: ${\n",
              "            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :\n",
              "                                    'n/a'} - `));\n",
              "    const percent = span('0% done');\n",
              "    li.appendChild(percent);\n",
              "\n",
              "    outputElement.appendChild(li);\n",
              "\n",
              "    const fileDataPromise = new Promise((resolve) => {\n",
              "      const reader = new FileReader();\n",
              "      reader.onload = (e) => {\n",
              "        resolve(e.target.result);\n",
              "      };\n",
              "      reader.readAsArrayBuffer(file);\n",
              "    });\n",
              "    // Wait for the data to be ready.\n",
              "    let fileData = yield {\n",
              "      promise: fileDataPromise,\n",
              "      response: {\n",
              "        action: 'continue',\n",
              "      }\n",
              "    };\n",
              "\n",
              "    // Use a chunked sending to avoid message size limits. See b/62115660.\n",
              "    let position = 0;\n",
              "    do {\n",
              "      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);\n",
              "      const chunk = new Uint8Array(fileData, position, length);\n",
              "      position += length;\n",
              "\n",
              "      const base64 = btoa(String.fromCharCode.apply(null, chunk));\n",
              "      yield {\n",
              "        response: {\n",
              "          action: 'append',\n",
              "          file: file.name,\n",
              "          data: base64,\n",
              "        },\n",
              "      };\n",
              "\n",
              "      let percentDone = fileData.byteLength === 0 ?\n",
              "          100 :\n",
              "          Math.round((position / fileData.byteLength) * 100);\n",
              "      percent.textContent = `${percentDone}% done`;\n",
              "\n",
              "    } while (position < fileData.byteLength);\n",
              "  }\n",
              "\n",
              "  // All done.\n",
              "  yield {\n",
              "    response: {\n",
              "      action: 'complete',\n",
              "    }\n",
              "  };\n",
              "}\n",
              "\n",
              "scope.google = scope.google || {};\n",
              "scope.google.colab = scope.google.colab || {};\n",
              "scope.google.colab._files = {\n",
              "  _uploadFiles,\n",
              "  _uploadFilesContinue,\n",
              "};\n",
              "})(self);\n",
              "</script> "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving data.csv to data.csv\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data=pd.read_csv('data.csv')\n",
        "data.head(10)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 474
        },
        "id": "7nvSh82LHIsh",
        "outputId": "4a062a59-3d99-4d54-ef8f-cb090affcaaf"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "         id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \\\n",
              "0    842302         M        17.99         10.38          122.80     1001.0   \n",
              "1    842517         M        20.57         17.77          132.90     1326.0   \n",
              "2  84300903         M        19.69         21.25          130.00     1203.0   \n",
              "3  84348301         M        11.42         20.38           77.58      386.1   \n",
              "4  84358402         M        20.29         14.34          135.10     1297.0   \n",
              "5    843786         M        12.45         15.70           82.57      477.1   \n",
              "6    844359         M        18.25         19.98          119.60     1040.0   \n",
              "7  84458202         M        13.71         20.83           90.20      577.9   \n",
              "8    844981         M        13.00         21.82           87.50      519.8   \n",
              "9  84501001         M        12.46         24.04           83.97      475.9   \n",
              "\n",
              "   smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \\\n",
              "0          0.11840           0.27760         0.30010              0.14710   \n",
              "1          0.08474           0.07864         0.08690              0.07017   \n",
              "2          0.10960           0.15990         0.19740              0.12790   \n",
              "3          0.14250           0.28390         0.24140              0.10520   \n",
              "4          0.10030           0.13280         0.19800              0.10430   \n",
              "5          0.12780           0.17000         0.15780              0.08089   \n",
              "6          0.09463           0.10900         0.11270              0.07400   \n",
              "7          0.11890           0.16450         0.09366              0.05985   \n",
              "8          0.12730           0.19320         0.18590              0.09353   \n",
              "9          0.11860           0.23960         0.22730              0.08543   \n",
              "\n",
              "   ...  texture_worst  perimeter_worst  area_worst  smoothness_worst  \\\n",
              "0  ...          17.33           184.60      2019.0            0.1622   \n",
              "1  ...          23.41           158.80      1956.0            0.1238   \n",
              "2  ...          25.53           152.50      1709.0            0.1444   \n",
              "3  ...          26.50            98.87       567.7            0.2098   \n",
              "4  ...          16.67           152.20      1575.0            0.1374   \n",
              "5  ...          23.75           103.40       741.6            0.1791   \n",
              "6  ...          27.66           153.20      1606.0            0.1442   \n",
              "7  ...          28.14           110.60       897.0            0.1654   \n",
              "8  ...          30.73           106.20       739.3            0.1703   \n",
              "9  ...          40.68            97.65       711.4            0.1853   \n",
              "\n",
              "   compactness_worst  concavity_worst  concave points_worst  symmetry_worst  \\\n",
              "0             0.6656           0.7119                0.2654          0.4601   \n",
              "1             0.1866           0.2416                0.1860          0.2750   \n",
              "2             0.4245           0.4504                0.2430          0.3613   \n",
              "3             0.8663           0.6869                0.2575          0.6638   \n",
              "4             0.2050           0.4000                0.1625          0.2364   \n",
              "5             0.5249           0.5355                0.1741          0.3985   \n",
              "6             0.2576           0.3784                0.1932          0.3063   \n",
              "7             0.3682           0.2678                0.1556          0.3196   \n",
              "8             0.5401           0.5390                0.2060          0.4378   \n",
              "9             1.0580           1.1050                0.2210          0.4366   \n",
              "\n",
              "   fractal_dimension_worst  Unnamed: 32  \n",
              "0                  0.11890          NaN  \n",
              "1                  0.08902          NaN  \n",
              "2                  0.08758          NaN  \n",
              "3                  0.17300          NaN  \n",
              "4                  0.07678          NaN  \n",
              "5                  0.12440          NaN  \n",
              "6                  0.08368          NaN  \n",
              "7                  0.11510          NaN  \n",
              "8                  0.10720          NaN  \n",
              "9                  0.20750          NaN  \n",
              "\n",
              "[10 rows x 33 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-7daa06ae-423c-4422-b1ef-4eaae967e213\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>id</th>\n",
              "      <th>diagnosis</th>\n",
              "      <th>radius_mean</th>\n",
              "      <th>texture_mean</th>\n",
              "      <th>perimeter_mean</th>\n",
              "      <th>area_mean</th>\n",
              "      <th>smoothness_mean</th>\n",
              "      <th>compactness_mean</th>\n",
              "      <th>concavity_mean</th>\n",
              "      <th>concave points_mean</th>\n",
              "      <th>...</th>\n",
              "      <th>texture_worst</th>\n",
              "      <th>perimeter_worst</th>\n",
              "      <th>area_worst</th>\n",
              "      <th>smoothness_worst</th>\n",
              "      <th>compactness_worst</th>\n",
              "      <th>concavity_worst</th>\n",
              "      <th>concave points_worst</th>\n",
              "      <th>symmetry_worst</th>\n",
              "      <th>fractal_dimension_worst</th>\n",
              "      <th>Unnamed: 32</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>842302</td>\n",
              "      <td>M</td>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.30010</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>...</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>842517</td>\n",
              "      <td>M</td>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.08690</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>...</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>84300903</td>\n",
              "      <td>M</td>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.19740</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>...</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>84348301</td>\n",
              "      <td>M</td>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.24140</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>...</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>84358402</td>\n",
              "      <td>M</td>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.19800</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>...</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>843786</td>\n",
              "      <td>M</td>\n",
              "      <td>12.45</td>\n",
              "      <td>15.70</td>\n",
              "      <td>82.57</td>\n",
              "      <td>477.1</td>\n",
              "      <td>0.12780</td>\n",
              "      <td>0.17000</td>\n",
              "      <td>0.15780</td>\n",
              "      <td>0.08089</td>\n",
              "      <td>...</td>\n",
              "      <td>23.75</td>\n",
              "      <td>103.40</td>\n",
              "      <td>741.6</td>\n",
              "      <td>0.1791</td>\n",
              "      <td>0.5249</td>\n",
              "      <td>0.5355</td>\n",
              "      <td>0.1741</td>\n",
              "      <td>0.3985</td>\n",
              "      <td>0.12440</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>844359</td>\n",
              "      <td>M</td>\n",
              "      <td>18.25</td>\n",
              "      <td>19.98</td>\n",
              "      <td>119.60</td>\n",
              "      <td>1040.0</td>\n",
              "      <td>0.09463</td>\n",
              "      <td>0.10900</td>\n",
              "      <td>0.11270</td>\n",
              "      <td>0.07400</td>\n",
              "      <td>...</td>\n",
              "      <td>27.66</td>\n",
              "      <td>153.20</td>\n",
              "      <td>1606.0</td>\n",
              "      <td>0.1442</td>\n",
              "      <td>0.2576</td>\n",
              "      <td>0.3784</td>\n",
              "      <td>0.1932</td>\n",
              "      <td>0.3063</td>\n",
              "      <td>0.08368</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>84458202</td>\n",
              "      <td>M</td>\n",
              "      <td>13.71</td>\n",
              "      <td>20.83</td>\n",
              "      <td>90.20</td>\n",
              "      <td>577.9</td>\n",
              "      <td>0.11890</td>\n",
              "      <td>0.16450</td>\n",
              "      <td>0.09366</td>\n",
              "      <td>0.05985</td>\n",
              "      <td>...</td>\n",
              "      <td>28.14</td>\n",
              "      <td>110.60</td>\n",
              "      <td>897.0</td>\n",
              "      <td>0.1654</td>\n",
              "      <td>0.3682</td>\n",
              "      <td>0.2678</td>\n",
              "      <td>0.1556</td>\n",
              "      <td>0.3196</td>\n",
              "      <td>0.11510</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>844981</td>\n",
              "      <td>M</td>\n",
              "      <td>13.00</td>\n",
              "      <td>21.82</td>\n",
              "      <td>87.50</td>\n",
              "      <td>519.8</td>\n",
              "      <td>0.12730</td>\n",
              "      <td>0.19320</td>\n",
              "      <td>0.18590</td>\n",
              "      <td>0.09353</td>\n",
              "      <td>...</td>\n",
              "      <td>30.73</td>\n",
              "      <td>106.20</td>\n",
              "      <td>739.3</td>\n",
              "      <td>0.1703</td>\n",
              "      <td>0.5401</td>\n",
              "      <td>0.5390</td>\n",
              "      <td>0.2060</td>\n",
              "      <td>0.4378</td>\n",
              "      <td>0.10720</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>84501001</td>\n",
              "      <td>M</td>\n",
              "      <td>12.46</td>\n",
              "      <td>24.04</td>\n",
              "      <td>83.97</td>\n",
              "      <td>475.9</td>\n",
              "      <td>0.11860</td>\n",
              "      <td>0.23960</td>\n",
              "      <td>0.22730</td>\n",
              "      <td>0.08543</td>\n",
              "      <td>...</td>\n",
              "      <td>40.68</td>\n",
              "      <td>97.65</td>\n",
              "      <td>711.4</td>\n",
              "      <td>0.1853</td>\n",
              "      <td>1.0580</td>\n",
              "      <td>1.1050</td>\n",
              "      <td>0.2210</td>\n",
              "      <td>0.4366</td>\n",
              "      <td>0.20750</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>10 rows × 33 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-7daa06ae-423c-4422-b1ef-4eaae967e213')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-7daa06ae-423c-4422-b1ef-4eaae967e213 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-7daa06ae-423c-4422-b1ef-4eaae967e213');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "o23pEkvQHZz2",
        "outputId": "c1f418d1-7636-4048-eacb-48f9d70c1192"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(569, 33)"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Mapping dignosis variable as o and 1"
      ],
      "metadata": {
        "id": "hUyN2ESLHnPt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data['diagnosis']=data['diagnosis'].map({'B':0,'M':1}).astype(int)\n",
        "data['diagnosis']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KwYjtfixHxHH",
        "outputId": "6b62e787-5022-4711-d21b-acdb9056eaef"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0      1\n",
              "1      1\n",
              "2      1\n",
              "3      1\n",
              "4      1\n",
              "      ..\n",
              "564    1\n",
              "565    1\n",
              "566    1\n",
              "567    1\n",
              "568    0\n",
              "Name: diagnosis, Length: 569, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Segragating dataset into x and y"
      ],
      "metadata": {
        "id": "4pwOQM3-IgIY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x=data.iloc[:,2:32].values\n",
        "x"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fx_mCNMdIm4I",
        "outputId": "5db1b6fd-c2da-4eb0-fffb-d86f4513c2d1"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,\n",
              "        1.189e-01],\n",
              "       [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,\n",
              "        8.902e-02],\n",
              "       [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,\n",
              "        8.758e-02],\n",
              "       ...,\n",
              "       [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,\n",
              "        7.820e-02],\n",
              "       [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,\n",
              "        1.240e-01],\n",
              "       [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,\n",
              "        7.039e-02]])"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y=data.iloc[:,1].values\n",
        "y"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xPJYosHpI7m2",
        "outputId": "cf17b019-1a32-4a90-e488-50de4226e928"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,\n",
              "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,\n",
              "       1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1,\n",
              "       0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,\n",
              "       0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,\n",
              "       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1,\n",
              "       1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,\n",
              "       0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,\n",
              "       0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1,\n",
              "       1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1,\n",
              "       0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0,\n",
              "       0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
              "       1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0,\n",
              "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,\n",
              "       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1,\n",
              "       1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,\n",
              "       1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1,\n",
              "       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,\n",
              "       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,\n",
              "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0,\n",
              "       0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,\n",
              "       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
              "       0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,\n",
              "       0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,\n",
              "       0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
              "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0])"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Splitting dataset into train and test data"
      ],
      "metadata": {
        "id": "wo0YXWImJT8_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)"
      ],
      "metadata": {
        "id": "tB9EFfq0JbNy"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Feature Scaling\n",
        "### We scale our data to make all our features contrubute equally to the results\n",
        "## fit_transform- This method calculates the mean and variance of each features present in our data\n",
        "## transform - This method transforming all the features using respective mean and variances. "
      ],
      "metadata": {
        "id": "Pjj3lA7EJwL-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "sc=StandardScaler()\n",
        "x_train=sc.fit_transform(x_train)\n",
        "x_test=sc.transform(x_test)"
      ],
      "metadata": {
        "id": "cyJFj0qtJ3cO"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Validation of various ML algorithms by their accuracy"
      ],
      "metadata": {
        "id": "tQoU7FTKLK-J"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
        "\n",
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.model_selection import StratifiedKFold"
      ],
      "metadata": {
        "id": "8lIz_czULk7r"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "models=[]\n",
        "models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))\n",
        "models.append(('CART',DecisionTreeClassifier(criterion='entropy')))\n",
        "models.append(('CART_RF',RandomForestClassifier(criterion='gini')))\n",
        "models.append(('naive bayes',GaussianNB()))\n",
        "models.append(('SVC',SVC(gamma='auto',kernel='poly')))\n",
        "models.append(('KNN',KNeighborsClassifier()))\n",
        "models.append(('LDA',LinearDiscriminantAnalysis()))"
      ],
      "metadata": {
        "id": "E9eg-ZzjNnML"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "names=[]\n",
        "result=[]\n",
        "res=[]\n",
        "\n",
        "for name,model in models:\n",
        "  kfold=StratifiedKFold(n_splits=10,random_state=None)\n",
        "  cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')\n",
        "  result.append(cv_results)\n",
        "  names.append(name)\n",
        "  res.append(cv_results.mean())\n",
        "  print('%s:,%f (%f)' % (name,cv_results.mean(),cv_results.std()))\n",
        "\n",
        "plt.ylim(.800,.999)\n",
        "plt.bar(names,res,color='blue',width=0.4)\n",
        "plt.title('Algorithm comparision')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 403
        },
        "id": "msRko-A8NbUR",
        "outputId": "0ed6b12e-c173-40ad-d020-c767c668551e"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "LR:,0.981285 (0.025173)\n",
            "CART:,0.917774 (0.033719)\n",
            "CART_RF:,0.957807 (0.032610)\n",
            "naive bayes:,0.941417 (0.027918)\n",
            "SVC:,0.887375 (0.037357)\n",
            "KNN:,0.964839 (0.018995)\n",
            "LDA:,0.957863 (0.020150)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYAAAAEJCAYAAACdePCvAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAfPElEQVR4nO3df7xVdZ3v8ddbFKwUUzkZAgoplSiKucVuMwpZFnIdQXQUwvzxcLRmBu9NsxuWFnH1Ot2p9NqlTAsJU5F0LO6Ug5VQ1tViowiioQcs+WF6TK1MU8HP/LG+B5fbfTiLc/bhnN16Px+P/Thrfb/f9d2ftc/e67PXd/3YigjMzKx8durtAMzMrHc4AZiZlZQTgJlZSTkBmJmVlBOAmVlJOQGYmZWUE4D1GEnzJF3WQ31Pl3TnNurHS9rQE8/drDp7zXLtrpF06Y6IyXqXfB2AdZekpcBhwNsj4qVc+TxgQ0RcsgNiCGBkRLSm+fHAdyJiaE8/t1mz8h6AdYuk4cDRQAAn7qDn3HlHPM9fE79mVo8TgHXXGcC9wDzgzG01lPQ/JD0haZOkf5AUkg5MdXtImi+pTdJvJV0iaadUd5akX0i6UtLvgVmp7Oep/mfpKR6Q9Lyk03LP+UlJT6XnPTtXPk/S1yTdkZb5haS3S7pK0rOSfi3p8G2sy8GSfiTpGUlPSvpMKh+Q+tiUHldJGpDqxkvakF6H9pgmS5oo6ZHU12dyzzFL0q2SbpH0J0n3STosVz9T0tpU95Ckk3J1nb1mSnVPSfqjpFWSDsm9Npfl+jpXUmuKb5GkfXN1Ienjkh6V9JykOZK0rfeB9R1OANZdZwA3pseHJe1Tr5GkCcCFwAeBA4HxNU2+CuwBvAMYl/o9O1d/FLAO2Ae4PL9gRByTJg+LiN0i4pY0//bU5xDgHGCOpD1zi54KXAIMAl4C7gHuS/O3Al/pYF12B34M/Aewb1qfn6TqzwLvBcaQDYuNTc/R7u3ArimmzwHXAacDR5DtSV0qaUSu/STgu8BewE3A9yTtkurWpmX2AL4AfEfS4CKvGfAh4BjgnWn5U4Hf11nXY4ErUv1g4LfAgppmJwBHAoemdh+u7cf6qIjww48uPYC/BV4BBqX5XwMX5OrnAZel6bnAFbm6A8mGjQ4E+gEvA6Ny9R8Dlqbps4DHa577LODnufkADszNjwdeBHbOlT0FvDcX23W5uvOBh3Pzo4HnOljvacD9HdStBSbm5j8M/KYmpn5pfvcU91G59suByWl6FnBvrm4n4Ang6A6eewUwqchrBhwLPEKWrHaqaZf/v30L+N+5ut3S/3x47nX/21z9QmBmb783/Sj28B6AdceZwJ0R8XSav4mOh4H2Bdbn5vPTg4BdyL5dtvst2bfkeu2L+n1EbM7Nv0C2AWv3ZG76xTrz+bZ5w8g29PXsyxvXY9/c/O8jYkvuOerFkX/eresdEa8CG9r7k3SGpBVp6OU54BCy1/INy9aKiLuA/wvMAZ6SdK2kgZ2tT0Q8T7ankP/f/C43XfsaWx/mBGBdIulNZLv74yT9TtLvgAuAw/Lj1DlPAPkzcoblpp8m+1a5f65sP2Bjbr4vna62nmyoqp5NvHE9NnXjuba+TumYyFBgk6T9yYaPZgB7R8RbgQeB/Pj7Nl+ziLg6Io4ARpENBX2qTrPXrY+ktwB78/r/jTUpJwDrqsnAFrKNx5j0OAi4m2z8vtZC4GxJB0l6M7D1PPP0jXghcLmk3dPG7ULgO9sRz5N0vFFutH8HBkv6RDrou7uko1LdzcAlklokDSIb59+e9ah1hKQpys7i+QTZsYp7gbeQbeDbANIB7kOKdirpSElHpeMJfwb+Arxap+nNZP+3Melg9v8CfhkRv+nGOlkf4QRgXXUmcH1EPB4Rv2t/kA0rTFfNaYcRcQdwNbAEaCXbiEG2QYNsDP7PZActf042nDR3O+KZBXw7DYec2sV1KiQi/gQcB/wd2fDHo8D7U/VlQBVYCawiO6jcnYvhvg+cBjwLfBSYEhGvRMRDwJfJDlw/SXbM4hfb0e9Asj2IZ8mGeH4P/Gtto4j4MVmyvo1sL+4AYGpXV8b6Fl8IZr1C0kFkQxYDasbpLZE0i+zA9um9HYv9dfIegO0wkk5KQyZ7Al8E/p83/ma9xwnAdqSPkZ2KuZbs+ME/9m44ZuXmISAzs5LyHoCZWUk11Q2iBg0aFMOHD+/tMMzMmsry5cufjoiW2vKmSgDDhw+nWq32dhhmZk1F0m/rlXsIyMyspJwAzMxKygnAzKyknADMzErKCcDMrKScAMzMSsoJwMyspJwAzMxKygnAzKyknADMzErKCcDMrKScAMzMSsoJwMyspJwAzMxKygnAzKyknADMzErKCcDMrKScAMzMSqqpfhKyO6TG9RXRuL7MzHqL9wDMzErKCcDMrKQKJQBJEyStkdQqaWad+v0l/UTSSklLJQ1N5e+XtCL3+IukyalunqTHcnVjGrtqZma2LZ0eA5DUD5gDHAdsAJZJWhQRD+WafQmYHxHflnQscAXw0YhYAoxJ/ewFtAJ35pb7VETc2phVMTOz7VFkD2As0BoR6yLiZWABMKmmzSjgrjS9pE49wCnAHRHxQleDNTOzximSAIYA63PzG1JZ3gPAlDR9ErC7pL1r2kwFbq4puzwNG10paUDBmM3MrAEadRD4ImCcpPuBccBGYEt7paTBwGhgcW6Zi4F3A0cCewGfrtexpPMkVSVV29raGhSumZkVSQAbgWG5+aGpbKuI2BQRUyLicOCzqey5XJNTgdsj4pXcMk9E5iXgerKhpjeIiGsjohIRlZaWlkIrZWZmnSuSAJYBIyWNkNSfbChnUb6BpEGS2vu6GJhb08c0aoZ/0l4BkgRMBh7c/vDNzKyrOk0AEbEZmEE2fPMwsDAiVkuaLenE1Gw8sEbSI8A+wOXty0saTrYH8dOarm+UtApYBQwCLuvWmpiZ2XZRNNF9DSqVSlSr1S4t61tBmFlZSVoeEZXacl8JbGZWUk4AZmYl5QRgZlZSTgBmZiXlBGBmVlJOAGZmJVWaXwQzs+bj07d7lvcAzMxKygnAzKyknADMzErKCcDMrKScAMzMSsoJwMyspHwaqPU4n8pnZdQM73vvAZiZlZQTgJlZSTkBmJmVlBOAmVlJOQGYmZVUoQQgaYKkNZJaJc2sU7+/pJ9IWilpqaShubotklakx6Jc+QhJv0x93iKpf2NWyczMiug0AUjqB8wBjgdGAdMkjapp9iVgfkQcCswGrsjVvRgRY9LjxFz5F4ErI+JA4FngnG6sh5mZbaciewBjgdaIWBcRLwMLgEk1bUYBd6XpJXXqX0eSgGOBW1PRt4HJRYM2M7PuK5IAhgDrc/MbUlneA8CUNH0SsLukvdP8rpKqku6V1L6R3xt4LiI2b6NPACSdl5avtrW1FQjXzMyKaNRB4IuAcZLuB8YBG4EtqW7/iKgAHwGuknTA9nQcEddGRCUiKi0tLQ0K18zMitwKYiMwLDc/NJVtFRGbSHsAknYDTo6I51LdxvR3naSlwOHAbcBbJe2c9gLe0KeZmfWsInsAy4CR6ayd/sBUYFG+gaRBktr7uhiYm8r3lDSgvQ3wN8BDERFkxwpOScucCXy/uytjZmbFdZoA0jf0GcBi4GFgYUSsljRbUvtZPeOBNZIeAfYBLk/lBwFVSQ+QbfD/JSIeSnWfBi6U1Ep2TOBbDVonMzMrQNFEt1esVCpRrVa7tGwz3Jnvr5Vfe+uqZn7v9KXYJS1Px2Jfx1cCm5mVlBOAmVlJOQGYmZWUfxHMrBN9aSzXrJG8B2BmVlJOAGZmJeUEYGZWUk4AZmYl5QRgZlZSTgBmZiXlBGBmVlJOAGZmJeUEYGZWUk4AZmYl5QRgZlZSTgBmZiXlBGBmVlJOAGZmJVUoAUiaIGmNpFZJM+vU7y/pJ5JWSloqaWgqHyPpHkmrU91puWXmSXpM0or0GNO41TIzs850mgAk9QPmAMcDo4BpkkbVNPsSMD8iDgVmA1ek8heAMyLiYGACcJWkt+aW+1REjEmPFd1cFzMz2w5F9gDGAq0RsS4iXgYWAJNq2owC7krTS9rrI+KRiHg0TW8CngJaGhG4mZl1T5EEMARYn5vfkMryHgCmpOmTgN0l7Z1vIGks0B9Ymyu+PA0NXSlpQL0nl3SepKqkaltbW4FwzcysiEYdBL4IGCfpfmAcsBHY0l4paTBwA3B2RLyaii8G3g0cCewFfLpexxFxbURUIqLS0uKdBzOzRinym8AbgWG5+aGpbKs0vDMFQNJuwMkR8VyaHwj8APhsRNybW+aJNPmSpOvJkoiZme0gRfYAlgEjJY2Q1B+YCizKN5A0SFJ7XxcDc1N5f+B2sgPEt9YsMzj9FTAZeLA7K2JmZtun0wQQEZuBGcBi4GFgYUSsljRb0omp2XhgjaRHgH2Ay1P5qcAxwFl1Tve8UdIqYBUwCLisUStlZmadU0T0dgyFVSqVqFarXVpWalwcTfSS9QnN/to3e/zNrJlf+74Uu6TlEVGpLfeVwGZmJeUEYGZWUkXOArI+oC/tTprZXwfvAZiZlZQTgJlZSTkBmJmVlBOAmVlJOQGYmZWUE4CZWUk5AZiZlZQTgJlZSTkBmJmVlBOAmVlJOQGYmZWUE4CZWUk5AZiZlZQTgJlZSTkBmJmVVKEEIGmCpDWSWiXNrFO/v6SfSFopaamkobm6MyU9mh5n5sqPkLQq9Xl1+nF4MzPbQTpNAJL6AXOA44FRwDRJo2qafQmYHxGHArOBK9KyewGfB44CxgKfl7RnWubrwLnAyPSY0O21MTOzworsAYwFWiNiXUS8DCwAJtW0GQXclaaX5Oo/DPwoIp6JiGeBHwETJA0GBkbEvZH9Kv18YHI318XMzLZDkQQwBFifm9+QyvIeAKak6ZOA3SXtvY1lh6TpbfVpZmY9qFEHgS8Cxkm6HxgHbAS2NKJjSedJqkqqtrW1NaJLMzOjWALYCAzLzQ9NZVtFxKaImBIRhwOfTWXPbWPZjWm6wz5zfV8bEZWIqLS0tBQI18zMiiiSAJYBIyWNkNQfmAosyjeQNEhSe18XA3PT9GLgQ5L2TAd/PwQsjogngD9Kem86++cM4PsNWB8zMyuo0wQQEZuBGWQb84eBhRGxWtJsSSemZuOBNZIeAfYBLk/LPgP8T7IksgyYncoA/gn4JtAKrAXuaNRKmZlZ55SdhNMcKpVKVKvVLi3byKsMeuMla+b4mzl2aP74m1kzv/Z9KXZJyyOiUlvuK4HNzErKCcDMrKScAMzMSsoJwMyspJwAzMxKygnAzKyknADMzErKCcDMrKR27u0AzKzn9KWLkazv8R6AmVlJOQGYmZWUE4CZWUk5AZiZlZQTgJlZSTkBmJmVlBOAmVlJOQGYmZWUE4CZWUk5AZiZlVShBCBpgqQ1klolzaxTv5+kJZLul7RS0sRUPl3SitzjVUljUt3S1Gd73dsau2pmZrYtnd4LSFI/YA5wHLABWCZpUUQ8lGt2CbAwIr4uaRTwQ2B4RNwI3Jj6GQ18LyJW5JabHhFd+5V3MzPrliJ7AGOB1ohYFxEvAwuASTVtAhiYpvcANtXpZ1pa1szM+oAiCWAIsD43vyGV5c0CTpe0gezb//l1+jkNuLmm7Po0/HOpVP++hZLOk1SVVG1raysQrpmZFdGog8DTgHkRMRSYCNwgaWvfko4CXoiIB3PLTI+I0cDR6fHReh1HxLURUYmISktLS4PCNTOzIglgIzAsNz80leWdAywEiIh7gF2BQbn6qdR8+4+Ijenvn4CbyIaazMxsBymSAJYBIyWNkNSfbGO+qKbN48AHACQdRJYA2tL8TsCp5Mb/Je0saVCa3gU4AXgQMzPbYTo9CygiNkuaASwG+gFzI2K1pNlANSIWAZ8ErpN0AdkB4bMitv5+0DHA+ohYl+t2ALA4bfz7AT8GrmvYWpmZWacUTfQ7b5VKJarVrp012uw/jdfM8Tdz7NDc8Tdz7NDc8fel2CUtj4hKbbmvBDYzKyknADOzknICMDMrKScAM7OScgIwMyspJwAzs5JyAjAzKyknADOzknICMDMrKScAM7OScgIwMyspJwAzs5JyAjAzKyknADOzknICMDMrKScAM7OScgIwMyspJwAzs5IqlAAkTZC0RlKrpJl16veTtETS/ZJWSpqYyodLelHSivS4JrfMEZJWpT6vlhr5A2pmZtaZThOApH7AHOB4YBQwTdKommaXAAsj4nBgKvC1XN3aiBiTHh/PlX8dOBcYmR4Tur4aZma2vYrsAYwFWiNiXUS8DCwAJtW0CWBgmt4D2LStDiUNBgZGxL2R/Sr9fGDydkVuZmbdUiQBDAHW5+Y3pLK8WcDpkjYAPwTOz9WNSENDP5V0dK7PDZ30CYCk8yRVJVXb2toKhGtmZkU06iDwNGBeRAwFJgI3SNoJeALYLw0NXQjcJGngNvp5g4i4NiIqEVFpaWlpULhmZrZzgTYbgWG5+aGpLO8c0hh+RNwjaVdgUEQ8BbyUypdLWgu8My0/tJM+zcysBxXZA1gGjJQ0QlJ/soO8i2raPA58AEDSQcCuQJuklnQQGUnvIDvYuy4ingD+KOm96eyfM4DvN2SNzMyskE73ACJis6QZwGKgHzA3IlZLmg1UI2IR8EngOkkXkB0QPisiQtIxwGxJrwCvAh+PiGdS1/8EzAPeBNyRHmZmtoMoOwmnOVQqlahWq11atpFXGfTGS9bM8Tdz7NDc8Tdz7NDc8fel2CUtj4hKbbmvBDYzKyknADOzknICMDMrKScAM7OScgIwMyspJwAzs5JyAjAzKyknADOzknICMDMrKScAM7OScgIwMyspJwAzs5JyAjAzKyknADOzknICMDMrKScAM7OScgIwMyspJwAzs5IqlAAkTZC0RlKrpJl16veTtETS/ZJWSpqYyo+TtFzSqvT32NwyS1OfK9LjbY1bLTMz60ynPwovqR8wBzgO2AAsk7QoIh7KNbsEWBgRX5c0CvghMBx4Gvi7iNgk6RCyH5YfkltuekR07Ud+zcysW4rsAYwFWiNiXUS8DCwAJtW0CWBgmt4D2AQQEfdHxKZUvhp4k6QB3Q/bzMy6q0gCGAKsz81v4PXf4gFmAadL2kD27f/8Ov2cDNwXES/lyq5Pwz+XSlLxsM3MrLsadRB4GjAvIoYCE4EbJG3tW9LBwBeBj+WWmR4Ro4Gj0+Oj9TqWdJ6kqqRqW1tbg8I1M7MiCWAjMCw3PzSV5Z0DLASIiHuAXYFBAJKGArcDZ0TE2vYFImJj+vsn4CayoaY3iIhrI6ISEZWWlpYi62RmZgUUSQDLgJGSRkjqD0wFFtW0eRz4AICkg8gSQJuktwI/AGZGxC/aG0vaWVJ7gtgFOAF4sLsrY2ZmxXWaACJiMzCD7Ayeh8nO9lktabakE1OzTwLnSnoAuBk4KyIiLXcg8Lma0z0HAIslrQRWkO1RXNfolTMzs44p2043h0qlEtVq184abeQh5t54yZo5/maOHZo7/maOHZo7/r4Uu6TlEVGpLfeVwGZmJeUEYGZWUk4AZmYl5QRgZlZSTgBmZiXlBGBmVlJOAGZmJeUEYGZWUk4AZmYl5QRgZlZSTgBmZiXlBGBmVlJOAGZmJeUEYGZWUk4AZmYl5QRgZlZSTgBmZiXlBGBmVlJOAGZmJVUoAUiaIGmNpFZJM+vU7ydpiaT7Ja2UNDFXd3Fabo2kDxft08zMelanCUBSP2AOcDwwCpgmaVRNs0uAhRFxODAV+FpadlSaPxiYAHxNUr+CfZqZWQ8qsgcwFmiNiHUR8TKwAJhU0yaAgWl6D2BTmp4ELIiIlyLiMaA19VekTzMz60E7F2gzBFifm98AHFXTZhZwp6TzgbcAH8wte2/NskPSdGd9AiDpPOC8NPu8pDUFYu6OQcDT22og9XAEXddp7NDc8Tdz7NDc8Tdz7NDc8Tcg9v3rFRZJAEVMA+ZFxJcl/RfgBkmHNKLjiLgWuLYRfRUhqRoRlR31fI3UzLFDc8ffzLFDc8ffzLFD78ZfJAFsBIbl5oemsrxzyMb4iYh7JO1KltW2tWxnfZqZWQ8qcgxgGTBS0ghJ/ckO6i6qafM48AEASQcBuwJtqd1USQMkjQBGAr8q2KeZmfWgTvcAImKzpBnAYqAfMDciVkuaDVQjYhHwSeA6SReQHRA+KyICWC1pIfAQsBn454jYAlCvzx5Yv67YYcNNPaCZY4fmjr+ZY4fmjr+ZY4dejF/ZdtrMzMrGVwKbmZWUE4CZWUmVNgFIer5O2SxJGyWtkPSQpGm9EVtNTG+XtEDSWknLJf1Q0jtT3Sck/UXSHrn24yX9Ia3DryV9SdLoNL9C0jOSHkvTP+69Nes7JH1c0hkN6GeepFMaEVMjSfqspNXpNi0rJH1e0hU1bcZIejhN7ybpG7n33FJJda/T2QGxP5+bnijpEUn7p8/qC5Le1kHbkPTl3PxFkmbtsMDrxJQry29nHpX0b7V3Qkj/j5A0oSfjK20C2IYrI2IM2ZXJ35C0S28FIknA7cDSiDggIo4ALgb2SU2mkZ1RNaVm0bvTOhwOnAAMjIgxqWwR8Kk0/0G6qDcSk6Thkl7MJej57f+fmv63K7lFxDURMb+rr0Vflq7LOQF4T0QcSnaR5hLgtJqmU4Gb0/Q3gWeAkek9dzbZad29RtIHgKuB4yPit6n4abITUOp5CZgiqVfj3oYr02dwJHALcJekllz9NODn6W+PcQLoQEQ8CrwA7NmLYbwfeCUirmkviIgHIuJuSQcAu5Hdh6numyQiXgRW8NrV1w3Ry4lpbWo/muz6kVPz/QOTgQHAY+lb752S3pTiPlfSMkkPSLpN0ptT+az0DfHdkn6VW8/hklal6SMk/TQlu8WSBncQ3wclVdM31RNy/dwt6b70eF8qny9pcu75bpQ0Sdn9sv41xbpS0sdS/WBJP0sJ7kFJR2/jdWo3GHg6Il4CiIinI+JnwLM13+pPBW5O76ujgEsi4tW0zGMR8YMCz9UjJB0DXAecEBFrc1VzgdMk7VVnsc1kZ9dcsANC7JaIuAW4E/gIbP18/T1wFnCcsuuqeoQTQAckvQd4NCKe6sUwDgGWd1A3leweSncD75K0T20DSXuSXXvxswbH1euJKZ1O/KsO+hgJzImIg4HngJNT+b9FxJERcRjwMNkFjPk+fw30V3bNCmTfkm9JexlfBU5JyW4ucHkHoQ0nu9fVfwWuSR/ep4DjIuI9qc+rU9tvkX3ISXtL7wN+kOL6Q0QcCRwJnJti+giwOCXAw8hew87cCQxLCelrksal8pvJ3kNIei/wTPrSczCwov107T5gAPA9YHL6/+Q9T/a/+O8dLDsHmJ7fE+3D7gPenabfBzyWkt1SsvdSj3ACeKMLJK0GfknHH/K+YBrZjfZeBW4j+8bQ7mhJD5BdXb04In7X4Ofu9cSUNqxHAf+RKz4a+CGwhdc+NMvJNsoAh6Rv4quA6WQbu1oLeW145DSy3fN3ka3zjyStIEtuQzsIbWFEvJo2puvIPtS7kF0nswr4LtkdcImIn5JdENlC9v+8LSI2Ax8CzkjP9Utgb7LXaxlwdhrLHh0Rf+rsdYqI54EjyO6n1UaW0M5K63WKpJ14/fBPX/MK8P+pSdY5VwNnStq9tiIi/gjMB/5bz4XXMPm7/Uwj+wyR/vbYMJATwBtdmb45ngx8qyd3vwpYTfbhfR1Jo8k2CD+S9BuyD3D+TXJ3+pZ7MHCOpDE7INZ2PZ2YDkgbxieBJyJiZa7ubmAi2Z5be/LewmsXPM4DZkTEaOALZFes17oFODUdz4i0IRewun24KiJGR8SHOoiv9sKaIBuGeJLsW3sF6J+rnw+cTjbOPjeVCTg/93wjIuLONHRzDNnrN08FD1xHxJaIWBoRnwdmACdHxHrgMWAc2Xv9ltR8NXCYslu29wWvkg1PjZX0mdrKiHgOuAn45w6Wv4osebylxyJsjMOBh9PrfjLwufTZ/iowoV6CawQngA6kK5yrwJm9GMZdwABld0QFQNKhZN96ZkXE8PTYF9hX0uvu+Jduwf0vwKcbHFdvJqb2YwAHAEdIOnE7lt0deCIN6Uyv1yDtdm8BLuW1jeIaoCUdUEXSLpLq7T0A/L2kndJQ2DvSsnuQJatXgY+SXf3ebh7wifTcD6WyxcA/5g5wv1PSW9L/98mIuI7sQO17OlthSe+SNDJXNAZoP4h6M3AlsC4iNuTWvwp8IY1Ftx/D6LFhiM5ExAtke3TTJdXbE/gK8DHq3NkgIp4h26vraA+i10k6mWyv72ayW+qsjIhh6bO9P9kXqZN64rnLnADeLGlD7nFhnTazgQvTbvIOl26ncRLZgcW1aWjqCmA82UHYvNtJY7o1rgGOkTS8gaH1emKKiKeBmWQHn4u6lGxI5RdA7Xhy3i1k38oXpud6GTgF+GLag1lBNk5bz+NkxybuAD4eEX8h+4GkM9Oy7wb+nFuPJ8mOR1yf6+ObZLdPuU/Sg8A3yDZu44EHJN1PNjz1fwqs827At5WdNbWSbPhpVqr7Llkyrh3++QeyA/qt6fnnkR3H6DVpQz4BuKQ26af3wu1kxwvq+TK9dxZTR9uZC9LB/EfJ3mvHRkQb2Rem2s/2bfTQMJBvBWFdImlfst3rI4C/AL8hG345KH+wTtJXyIY/fglcFBHtZ8a8iewHgv4mIn4jaR7w7xFx6zaec3hqc0iaF9nGeAbZt+qt/TcLZWcirSI7TfMPvR2PlYsTgFkvkfRBsjOBroyIq3o7HisfJwAzs5Jq1C+CmTVMOph8Q03xSxHRK7cjMPtr5T0AM7OSKvNZQGZmpeYEYGZWUk4AZmYl5QRgZlZS/wmEMoAJyQTC6AAAAABJRU5ErkJggg==\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Training and Prediction using algorithm with high accuracy"
      ],
      "metadata": {
        "id": "JnfqFxIuVfF0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "model=LinearDiscriminantAnalysis()\n",
        "model.fit(x_train,y_train)\n",
        "y_pred=model.predict(x_test)\n",
        "print(np.column_stack((y_pred,y_test)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "66lUjeWSVeTR",
        "outputId": "e23b0770-6d2a-4a7b-b1ae-df2d0b32773c"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [0 1]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 1]\n",
            " [0 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 1]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [1 1]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import accuracy_score\n",
        "print('Accuracy of linear discriminant analysis model is :{0}% '.format(accuracy_score(y_pred,y_test)*100))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aCZL4UJAZCgj",
        "outputId": "662f3bad-9b3e-417c-96c3-d959328ba2a2"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy of linear discriminant analysis model is :97.2027972027972% \n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Conclusion\n",
        "### As we have train our breast cancer data with various ML classification algorithms out of that logistic regression algorithms gives best accuracy ammong them.\n",
        "### so,we have to use logistic regression algorithm for further prediction "
      ],
      "metadata": {
        "id": "Ha9YRkFXVXOO"
      }
    }
  ]
}
