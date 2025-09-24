# Timeseries-Noise-Regression-MFL
# 누설자속탐상(MFL) 공정 센서 데이터 기반의 시계열 노이즈 회귀 예측

## 프로젝트 개요
### 누설자속탐상(MFL) 공정의 데이터 기반 최적화: 미래 신호값의 추세 예측을 통한 선제적 위험 감지

누설자속탐상(MFL) 검사 공정의 핵심은 장비의 정밀도를 유지하는 교정 작업입니다.

<img width="888" height="555" alt="image" src="https://github.com/user-attachments/assets/69b5bf2b-0d52-4126-917a-751d56b22561" />

<누설자속탐상(MFL) 공정 흐름도로, 위와 아래 각각 5개의 센서를 통해 강철바 검사를 수행합니다.>

<img width="555" height="555" alt="image" src="https://github.com/user-attachments/assets/00c6ddfd-1de7-47e1-916b-18a31f9ca189" />

<누설자속탐상기>

"LOT(각 강종의 여러개의 강철바의 묶음) 변경 시 교정"이라는 현재 규칙은 생산 현장의 현실과 맞지 않아 자주 무시되며, 이는 측정값의 신뢰도 하락과 잠재적 불량품 발생의 원인이 되기도 합니다. 규칙을 절대적으로 고수를 하게 된다면 생산성이 저하되고, 무시를 하면 품질이 떨어지는 리스크가 증가하는 딜레마에 빠져있는 상황입니다. 

특히, 실제 위험 신호는 급격하게 튀는 단발성 스파이크나 급격히 하락하는 신호값이 아닌, 장비 상태가 점진적으로 악화되면서 나타나는 신호값의 완만한 우상향 추세가 위험 신호입니다. 해당 프로젝트는 이 문제에 집중하여, 미래(t+10) 시점의 강철바 노이즈 값을 예측하는 회귀 모델링 실험을 진행하였습니다. 

이는 단순히 미래의 한 지점에 대한 예측을 넘어, 현재(t) 시점과 예측된 미래(t+10) 시점 사이의 기울기를 계산할 수 있는 근거를 제공합니다. 또한 지속적으로 강철바가 들어올 때마다 미래 값에 대한 예측을 통해 만들어지는 신호값의 추세를 정량적으로 파악하여, 위험이 임계치에 도달하기 전에 선제적으로 대응할 수 있는 가능성을 제시합니다. 

최종적으로 개발된 LightGBM 회귀 분석기 모델은 미래(t+10) 시점의 노이즈 수준을 평균 R²: 0.728의 신뢰도와 MAE: 0.0065(종속변수 평균: 0.0745)의 절대 평균 오차값으로 예측을 함으로써, 시계열 데이터 기반의 추세 예측이 어떻게 미래의 위험을 감지하고, 교정 시점 판단의 근거가 될 수 있는지에 대한 잠재력의 가능성을 입증하였습니다. 

## 목차

## 1. 데이터 소개 및 EDA
해당 프로젝트에 사용된 데이터는 누설자속탐상(MFL) 공정에서 10개의 채널의 센서에서 수집된 시계열 데이터입니다. 

<img width="888" height="456" alt="image" src="https://github.com/user-attachments/assets/97373daf-f1c3-43a0-b6d0-30677cec6015" />

<센서의 검출 감도 대 결함 길이의 특성>



<img width="555" height="555" alt="image" src="https://github.com/user-attachments/assets/155b5c59-945d-48e4-b118-b9274be99a2b" />

<실제 강철바가 누설자속탐상기에 투입되어 검사가 진행되는 공정 모습으로, 이 과정에서 10개 채널의 시계열 신호가 수집됩니다.>


## 데이터의 핵심 구조적 특성
각 데이터 파일은 하나의 강철바에 해당하며, 기업측과 보안 서약으로 인해 원본 데이터를 직접 공개할 수 없는 점 양해 부탁드립니다.

1-1. 미시적(Micro) 복접성: 단일 강철바 내부의 복잡한 신호값

하나의 강철바 내부에서 수집된 10개 채널의 원본 신호 값은 아래와 같이 매우 변동성이 크고 복잡한 패턴을 보입니다. 

<img width="1592" height="790" alt="1bar" src="https://github.com/user-attachments/assets/78065829-dfa5-4645-96e3-327d85d03e1e" />
<하나의 강철바에 대한 데이터>

1-2. 거시적(Macro) 단절성: LOT 변경 시 발생하는 신호 리셋

해당 데이터의 도전 과제의 핵심은 거시적 구조에 있습니다. 아래 그래프는 LOT가 변경이 될 때의 신호 변화를 보여줍니다. LOT가 변경될 때마다(강종 변경, 장비 교정이 규칙에 따라 정상적으로 이루어졌을 경우) 이전 시계열 패턴이 완전히 단절이 되고, 신호값이 새로운 기준선에서 시작되는 것을 확인 할 수 있습니다.

<img width="1489" height="690" alt="LOT star   end" src="https://github.com/user-attachments/assets/b66d7957-205d-47c2-98c9-67b67d60542c" />
<LOT 변경시 발생하는 신호값의 변화>

1-3. 데이터 병합 및 구조 

원본 85,841개 강철바 파일에서 여러 누락 또는 결측 파일을 제거하여 최종적으로 약 85,377개의 강철바 데이터를 확보하였습니다. 
확보된 데이터들을 시간 순서를 기준으로 메타 데이터에 기재된 정보들과 같이 하나의 데이터 프레임으로 통합을 하였습니다. 본래 데이터는 85,377의 행을 가지지만 추후 진행되는 실험에서 종속변수 생성 과정에서 로직 특성상 통합된 데이터의 행이 드랍이 되면서 행의 개수가 줄어들게 됩니다. 


## 2. t+1 시점 신호값 회귀 예측 모델링 실험

2-1. 종속변수 설정 근거

본래 프로젝트는 t+1 시점의 신호값을 예측하는 것에서부터 시작이 되었습니다. 

다단계 예측(Multi-step Ahead Forecast)의 특성
- 불확실성 증가: 예측 시점이 t+1 에서 t+h(h>1)로 멀어질수록 현재(t) 시점의 정보 영향력은 자연스럽게 감소합니다. 또한 t 시점과 t+h 시점 사이의 수많은 잠재적 변수를 모두 고려해야 하기 때문에 예측의 불확실성은 필연적으로 크게 증가합니다.
- 정밀도 저하: 이로 인해 t+1 예측에 비해 예측치의 신뢰 구간이 훨씬 넓어지며, 예측의 정밀도 또한 필연적으로 저하됩니다.

위의 다단계 예측의 특성이 갖는 리스크 때문에 초기 실험은 t+1을 종속변수로 설정하여 진행하였습니다.

2-2. t+1 모델링
사용 모델
1. `ElasticNet`
   -  빠르고 해석 가능한 선형 모델로 데이터가 선형적으로 얼마나 예측이 가능한지 확인
   -  문제의 기본적인 예측 가능성 타진
   -  회귀 계수를 통해 피처의 영향력을 직관적으로 확인 가능
2. `RandomForestRegressor`
   - 안정적 앙상블 모델, 성능 기준점 설정
   - 데이터의 기본적인 비선형 관계를 잡기 위함

### 사용 피처

### **정적 피처**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `STEEL_TYPE` | 강종 | `원본 메타데이터 값` |
| `SIZE` | 강철바 사이즈 | `원본 메타데이터 값` |
| `LINE_SPEED` | 검사 속도 | `원본 메타데이터 값`|
| `BAR_datetime` | 날짜와 검사 시간 | `원본 메타데이터 값` |

*생성이유*: 정적 피처는 각 검사의 가장 기본적인 환경 정보를 나타내며 특히 `STEEL_TYPE`과 `SIZE`는 LOT 단위로 변경되며 신호의 기준선에 큰 영향을 미치기 때문에, 추후에 모델에 입력을 했을 때 모델이 LOT별로 각기 다른 검사 환경을 구분하고 이해하는데 필요한 정보가 될 것입니다.  
    
### **내부 신호 피처**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `sensor_mean` | 채널 신호값의 평균 | `df['sensor_col'].mean()` |
| `sensor_std` | 채널 신호값의 표준편차 | `df['sensor_col'].std()` |
| `sensor_median` | 채널 신호값의 중위수 | `df['sensor_col'].median()`|
| `event_count_h` | 강철바의 High_level 결함 총 발생 횟수 | `df[h_envnt_col].sum()` |
| `event_count_l` | 강철바의 Law_level 결함 총 발생 횟수 | `df[l_envnt_col].sum()` |
| `sensor_peak_count` | 신호가 상단 임계값을 넘어선 횟수(피크 발생 횟수) | `(signal_original > upper_band_aligned).sum()` |

*생성이유*: 내부 신호 피처는 하나의 강철바를 통과하는 전체 신호의 분포 특성을 평균, 표준편차, 중위수 등의 하나의 통계 값으로 요약을 합니다. 해당 통계값들은 신호의 전반적인 수준과 안정성을 나타냅니다. 결함 발생 횟수와 피크 값 피처는 모델에게 순간적으로 튀는 값의 빈도를 알려줄 수 있습니다. 

### **미시적 시계열 rolling 파생변수**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `micro_sensor_rolling_std_mean_12` | 채널별 신호값의 이동 표준편차를 평균으로 집계, window=12 | `df[sensor_col].rolling(window=12, min_periods=1).std().mean()` |
| `micro_sensor_rolling_mean_std_12` | 채널별 신호값의 이동 평균을 표준편차로 집계, window=12 | `df[sensor_col].rolling(window=12, min_periods=1).mean().std()` |
| `micro_sensor_rolling_std_std_12` | 채널별 신호값의 이동 표준편차를 평균으로 집계, window=12 | `df[sensor_col].rolling(window=12, min_periods=1).std().std()` |
| `micro_sensor_rolling_std_mean_49` | 채널별 신호값의 이동 표준편차를 평균으로 집계, window=49 | `df[sensor_col].rolling(window=49, min_periods=1).std().mean()` |
| `micro_sensor_rolling_mean_std_49` | 채널별 신호값의 이동 평균을 표준편차로 집계, window=49 | `df[sensor_col].rolling(window=49, min_periods=1).mean().std()` |
| `micro_sensor_rolling_std_std_49` | 채널별 신호값의 이동 표준편차를 표준편차로 집계, window=49 | `df[sensor_col].rolling(window=49, min_periods=1).std().std()` |

*생성이유*: EDA과정에서 하나의 강철바 내부의 신호값을 나타내는 시각화 그래프에서 확인을 했듯이, 강철바 내부의 신호는 변동성이 큽니다. rolling 파생 변수는 window를 이동시키면서 통계량을 계산하여, 단순한 통계값으로는 알 수 없는 신호의 국소적인 변화와 변동성을 포착합니다. 또한 `std().mean()`, `mean().std()`, `std().std()` 를 사용하여 이중으로 집계를 하면서 단순 변동성이 아닌 변동성의 변화 양상이라는 정보들을 모델에게 알려주고자 하였습니다. window_size의 경우에는 12와 49로 나누어 단기적 변화를 모두 학습하도록 설계하였습니다. 

### **미시적 시계열 ewm 파생변수**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `micro_sensor_ewm_12_mean` | 채널별 신호값의 지수가중평균의 최종값, span=12 | `df[sensor_col].ewm(span=12, min_periods=1).mean().iloc[-1]` |
| `micro_sensor_ewm_12_std` | 채널별 신호값을 지수가중표준편차의 최종값, span=12 | `df[sensor_col].ewm(span=12, min_periods=1).std().iloc[-1]` |
| `micro_sensor_ewm_49_mean` | 채널별 신호값을 지수가중평균의 최종값, span=49 | `df[sensor_col].ewm(span=49, min_periods=1).mean().iloc[-1]` |
| `micro_sensor_ewm_49_std` | 채널별 신호값을 지수가중표준편차의 최종값, span=49 | `df[sensor_col].ewm(span=49, min_periods=1).std().iloc[-1]` |

*생성이유*: rolling은 과거 데이터에 동일한 가중치를 부여하지만, ewm은 최신 데이터에 더 큰 가중치를 줍니다. `iloc[-1]`을 통해 ewm 시계열의 최종값을 추출을 했던 이유는 가중값이 최대한 반영이된 최종 상태 값을 포착하기 위함이였습니다. 이는 강철바의 전반적인 상태를 하나의 값으로 효과적으로 요약하는 전략이 될 수 있습니다. 

2-3. 1차 모델링: 정적 피처 + 내부 신호 피처

1번 타자 모델링
- 가설: 내부 신호 피처와 정적 피처를 통해 대략적인 선형관계를 모델이 잡아낼 수 있을 것 이다.  
- 피처: 정적 피처 + 내부 신호 피처
- 알고리즘: `ElasticNet`
- 모델 파라미터: `alpha: 0.1, l1_ratio: 0.5`
- 평가지표:
  - `r2_score: -0.2304`
  - `mae: 0.0181`
  - `mse: 0.0006`
- 결론: 결정계수가 약 -0.2를 보이면서, 현재 피처들로는 데이터의 패턴을 선형적으로 설명할 수 없음을 확인했습니다. 해당 결과는 단순히 피처의 부족으로 인한 문제일 수도 있겠지만, 데이터가 가진 복잡한 비선형 관계를 선형 모델이 포착하지 못하는 한계일 가능성이라는 것 또한 있다고 생각을 했습니다. 따라서 다음 실험은 동일한 피처를 사용하면서, 비선형 관계를 학습할 수 있는 트리 기반 모델을 사용하여 가설을 검증해보기로 했습니다.

2번 타자 모델링
- 가설: 1번 타자 모델링과 피처셋은 동일하나 선형 모델에서 비선형 모델로 바뀌었기 때문에 모델이 데이터의 비선형 관계를 학습하여 기본적인 예측값을 뱉어낼 것 같다.
- 피처: 정적 피처 + 내부 신호 피처
- 알고리즘: `RandomForestRegressor`
- 모델 파라미터: X
- 평가지표:
  - `r2_score: -0.0785`
  - `mae: 0.0177`
  - `mse: 0.0007`
- 결론: 피처는 이전 실험과 동일하게 하여, 모델만 회귀 계수 모델에서 트리 기반 모델로 변경하여 예측을 시도해 보았으나, 여전히 결정계수가 음수를 띄면서 기존의 정적 피처와 기본 통계 값들이 대부분인 내부 신호 피처들로는 예측이 불가능하다는 결론을 내리게 되었습니다.
다음 실험은 추가적인 미시적인 시계열 파생변수들을 추가하여 진행합니다.

2-3. 2차 모델링: 정적 피처 + 내부 신호 피처 + 미시적 rolling 파생변수 + 미시적 ewm 파생변수

1번 타자 모델링
- 가설: 미시적인 rolling 파생변수와 미시적인 ewm 파생변수가로 모델이 데이터에서 선형적인 관계를 잡아내면서 예측이 될 거 같다.
- 피처: 정적 피처 + 내부 신호 피처 + 미시적 rolling 파생변수 + 미시적 ewm 파생변수
- 알고리즘: `ElasticNet`
- 파라미터: `alpha: 0.1, l1_ratio: 0.5`
- 평가지표:
  - `정확한 수치들은 유실되었으나 결정계수가 여전히 음수를 보였음`
- 결론: 실험노트를 작성하는 과정에서 `ElasticNet` 모델링 평가지표에 대한 정확한 점수는 유실되었지만 결정계수가 여전히 음수를 띄었습니다. 해당 시계열 회귀 예측 문제는 피처의 양만으로는 해결되지 않는 비선형성을 가지고 있음을 확인하여 추가적으로 진행되는 모든 모델링 실험에서는 선형 모델을 완전히 배제하고 비선형 관계 학습에 강점을 가진 트리 기반 모델만을 사용하기로 결정하였습니다.

2번 타자 모델링
- 가설: 시적인 rolling 파생변수와 미시적인 ewm 파생변수가로 선형 모델이 잡아내지 못했던 비선형적인 관계를 트리 모델이 학습을 하여 예측에 성능을 보일 것 같다.
- 피처: 정적 피처 + 내부 신호 피처 + 미시적 rolling 파생변수 + 미시적 ewm 파생변수
- 알고리즘: `RandomForesetRegressor `
- 파라미터: X
- 평가지표:
  - `r2_score: 0.6861`
  - `mae: 0.0093`
  - `mse: 0.0001`
- 시계열 교차검증:
  - `Fold 1/5: R²: 0.6203, MAE: 0.0106, MSE: 0.0004`
  - `Fold 2/5: R²: 0.8725, MAE: 0.0071, MSE: 0.0001`
  - `Fold 3/5: R²: 0.6785, MAE: 0.0082, MSE: 0.0001`
  - `Fold 4/5: R²: 0.8074, MAE: 0.0076, MSE: 0.0001`
  - `Fold 5/5: R²: 0.6215, MAE: 0.0092, MSE: 0.0001`
  - `Average: R²: 0.7200 (+/- 0.1022), MAE: 0.0085 (+/- 0.0013), MSE: 0.0002 (+/- 0.0001)`
- 결론: `RandomForesetRegressor `로 예측을 한 초기 모델링의 평가지표는 예상과 다르게 이전 보다 매우 높게 측정이 되어 누수가 발생을 했는지 확인을 해보기 위해 5개의 폴드로 나눠 폴드가 누적이 되는 방식으로 시계열 교차검증을 진행을 하여 평가지표를 뽑아 보았는데 폴드마다 약간의 편차를 보이기는 하지만 여전히 점수가 나오는 것을 확인 했습니다. 또한 결정계수 뿐만 아니라 평균 절대 오차 값 또한 종속변수의 평균값(0.0745)을 고려했을 때 꽤 높게 나온 점 또한 눈여겨 볼만합니다.
미시적인 rolling, ewm 파생변수가 강력한 변수인 것임은 예상은 했지만 예상과 다르게 과하다 싶을 정도의 점수가 나와 추가적인 검증이 필요할 것으로 보입니다.
종속변수가 `t+1`인 바로 다음 시퀀스를 예측하는 것이기 때문에 현재 시점인 `t`와의 강력한 자기 상관 또한 의심을 하게 되는 계기가 되었습니다.

3번 타자 모델링
- 가설: 시적인 rolling 파생변수와 미시적인 ewm 파생변수가로 `RandomForesetRegressor`보다 성능이 뛰어 나다고 알려진 `XGBoostRegressor`를 사용하여 얼마나 더 높은 점수를 보이는지 확인하면서, 해당 모델링의 결과에 상관없이 종속변수의 재설정 검토가 필요할 듯함.
- 피처: 정적 피처 + 내부 신호 피처 + 미시적 rolling 파생변수 + 미시적 ewm 파생변수
- 알고리즘: `XGBoostRegressor`
- 평가지표:
  - `r2_score: 0.6674`
  - `mae: 0.0097`
  - `mse: 0.0001`
- 시계열 교차검증:
  - `Fold 1/5: R²: 0.3209, MAE: 0.0116, MSE: 0.0007`
  - `Fold 2/5: R²: 0.8628, MAE: 0.0073, MSE: 0.0001`
  - `Fold 3/5: R²: 0.6586, MAE: 0.0082, MSE: 0.0001`
  - `Fold 4/5: R²: 0.7944, MAE: 0.0080, MSE: 0.0001`
  - `Fold 5/5: R²: 0.5788, MAE: 0.0097, MSE: 0.0002`
  - `Average: R²: 0.6431 (+/- 0.1894), 0.0090 (+/- 0.0015), 0.0002 (+/- 0.0002)`
- 피처 중요도
<img width="1106" height="682" alt="XGB1" src="https://github.com/user-attachments/assets/c0e1dbc5-9011-4d2d-bf47-6e9148624e3d" />


- SHAP value
<img width="803" height="940" alt="XGB2" src="https://github.com/user-attachments/assets/9a875da4-4e01-4f83-a8bb-38049020ebb0" />


결론: 분명 `XGBoostRegressor`가 `RandomForesetRegressor` 보다 더 높은 평가지표 점수를 반환해 낼 것이라고 가설을 설정하고 모델링 실험을 진행하였지만, 예상과 다르게 `RandomForesetRegressor`의 점수를 넘지 못했습니다. 해당 실험을 통해 `XGBoostRegressor`가 항상 뛰어난 성능을 보장한다는 것은 아님을 알 수 있었으며, 새롭게 `XGBoostRegressor`의 피처 중요도와 SHAP value 시각화 그래포 또한 뽑아 보았는데, 제일 처음의 정적인 피처와 내부 신호값으로만 모델링을 진행을 했을 때는 해당 피처들로는 예측이 불가능 하다는 결론을 내렸지만 미시적인 rolling, ewm 파생변수를 투입을 하고 보니 기존의 기본적인 통계값들이 중요도 부분에서 제일 상위 부분에 위치하는 것을 확인 할 수 있었습니다. 이는 새롭게 추가된 미시적 파생 변수들이 단독으로 강력한 예측력을 가졌다기보다, 기존의 기본 통계 피처들과 상호작용하며 그 피처들이 가진 포텐을 끌어내는 역할을 했음을 의미합니다. 이 분석을 통해, 단순히 개별적인 중요도가 낮은 피처라도 다른 변수와의 상호작용 관계 속에서 피처의 중요도가 향상이 될 수 있음을 확인했습니다.


## 3. 문제 정의의 진화, t+10 시점 신호값 회귀 예측 모델링 실험

3-1. 종속변수 설정 근거

프로젝트 초기에는 t+1 시점의 노이즈를 예측하는 것을 목표로 설정했습니다. 하지만 높은 예측 성능에도 불구하고, 문제 정의 자체에 대한 근본적인 의문을 가지게 되어 자기상관성 분석을 수행했습니다.
<img width="1389" height="989" alt="image" src="https://github.com/user-attachments/assets/80e0caba-fc6a-40b7-a885-72c2e80cf605" />
<ACF와 PACF 그래프>

자기 상관성 분석 결과, ACF는 매우 완만하게 감소하고 PACF는 Lag 1에서 급격히 절단되는 패턴이 확인 되었습니다.이는 타겟 변수가 강력한 추세와 자기상관성을 가지면서, 특히 현재 값`t`이 바로 다음 값인 `t+1`을 예측하는 데 결정적인 정보임을 의미합니다. 즉, `t+1` 예측은 모델의 성능이나 피처가 뛰어나다기 보다는, 문제 자체가 가진 특성으로 인해 모델이 풀기 쉬운 문제였던 것이라고 결론을 내릴 수 있겠습니다. 

이러한 단기 예측은 실질적인 비즈니스 가치를 제공하기 어렵다고 판단을 하여, 프로젝트의 목표를 단순 예측을 넘어 이보다 더 의미 있는 사전 경고 시스템을 구축하는 방향으로 재정의를 하기로 했습니다. 이에 따라 종속변수를 `t+1`에서 더 먼 미래인 `t+10`으로 전환을 하였습니다. `t+10`을 타겟으로 설정함으로써, 모델이 단순히 바로 직전 시점 `t`의 값을 복사하는 쉬운 학습과 예측을 하는 것을 방지하고, 데이터에 내재된 더 복잡하고 장기적인 패턴을 학습하도록 유도하는 효과도 있습니다 앞서 설명드린 다단계 예측이 갖는 리스크에도 불구하고 `t+10`을 종속 변수로 선택한 이유는, 예측 성공 시 실제 공정에서 `t+1`에 비해 압도적으로 긴 대응 시간을 확보하여 실질적인 예지 보전의 가능성을 열 수 있기 때문입니다. 

종속변수 특성
- `df['sensor_A1_y'].mean() = 0.0745` 종속변수의 평균 값의 경우 전체 채널의 종속변수 평균과 A1 센서의 종속변수 평균 값은 소수점 4자리 수 까지는 값이 같습니다. 
- 해당 실험은 우선적으로 10개의 센서중 A1의 센서를 대상으로 실험을 진행하였습니다.
- 종속변수가 `right_skew` 형태를 띄어 로그 변환을 해주어 실험을 진행하였습니다. 원래 `right_skew` 형태를 띈다고 해서 무조건적으로 로그변환을 해야하는 것은 아니지만, 두 변수를 따로 해서 실험을 진행하였을 때 로그변환된 종속변수를 대상으로 실험을 했을 때 예측 성능이 더 좋게 나와서 로그 변환을 하여 계속 실험을 진행하기로 하였습니다. 
<img width="1589" height="616" alt="image" src="https://github.com/user-attachments/assets/1a8c78cf-551f-4cae-8e9d-683bb5a82654" />
<로그변환 전(왼쪽)과 로그 변환 후(오른쪽)의 그래프>

3-2. 실험 계획
아래는 `t+10` 모델링 실험에 대한 계획입니다.
1. 단일 채널 우선 검증
   - 팀의 최종 목표는 모든 채널에 대한 예측 모델 개발이지만, 신속한 반복 실험과 가설 검증을 위해 우선 대표 채널 하나(sensor_A1)를 선정했습니다. 해당 단일 채널에 모든 방법론을 적용하여 모든 피처셋과 최적의 모델링을 선정한 이후 이를 나머지 채널로 확장하는 접근법을 선택했습니다.
  
2. 점진적 피처 엔지니어링
   - 피처가 추가가 될 때마다 모델의 성능이 얼마나 향상이 되는지를 검증하기 위하여, 기존의 피처셋에서 미시적, 거시적 시계열 피처 그룹을 순차적으로 추가하며 각 단계별로 성능의 변화를 측정하고 기록합니다.
  
3. 모델 전략
   - 비선형 모델 집중: 초기 `t+1` 실험에서 선형 모델의 한계를 확인했기 때문에, `t+1`에 비해 난이도가 더욱 향상된 `t+10` 예측에서는 비선형 관계 학습에 장점을 가진 트리 기반 모델만을 사용합니다.
   - XGBoost 우선 탐색: 초기 피처의 상호작용을 파악하기 위해 특정 중요 피처에 집중하는 `LightGBM`보다 다양한 피처를 비교적 균일하게 활용하는 `XGBoost`를 우선적으로 사용합니다.

4. 최종 모델 선정
   - Optuna를 사용한 하이퍼파라미터 최적화를 통해 `XGBoost`와 `LightGBM`의 성능을 극한으로 끌어올린 후, 최종적으로 가장 뛰어난 모델을 선정합니다. 선정된 모델을 추후에 10개의 통합된 채널 실험에 적용합니다.

3-3. 사용피처

### **정적 피처**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `STEEL_TYPE` | 강종 | `원본 메타데이터 값` |
| `SIZE` | 강철바 사이즈 | `원본 메타데이터 값` |
| `LINE_SPEED` | 검사 속도 | `원본 메타데이터 값`|
| `BAR_datetime` | 날짜와 검사 시간 | `원본 메타데이터 값` |

*생성이유*: 정적 피처는 각 검사의 가장 기본적인 환경 정보를 나타내며 특히 `STEEL_TYPE`과 `SIZE`는 LOT 단위로 변경되며 신호의 기준선에 큰 영향을 미치기 때문에, 추후에 모델에 입력을 했을 때 모델이 LOT별로 각기 다른 검사 환경을 구분하고 이해하는데 필요한 정보가 될 것입니다.  

### **내부 신호 피처**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `sensor_mean` | 채널 신호값의 평균 | `df['sensor_col'].mean()` |
| `sensor_std` | 채널 신호값의 표준편차 | `df['sensor_col'].std()` |
| `sensor_median` | 채널 신호값의 중위수 | `df['sensor_col'].median()`|
| `event_count_h` | 강철바의 High_level 결함 총 발생 횟수 | `df[h_envnt_col].sum()` |
| `event_count_l` | 강철바의 Law_level 결함 총 발생 횟수 | `df[l_envnt_col].sum()` |
| `sensor_peak_count` | 신호가 상단 임계값을 넘어선 횟수(피크 발생 횟수) | `(signal_original > upper_band_aligned).sum()` |

*생성이유*: 내부 신호 피처는 하나의 강철바를 통과하는 전체 신호의 분포 특성을 평균, 표준편차, 중위수 등의 하나의 통계 값으로 요약을 합니다. 해당 통계값들은 신호의 전반적인 수준과 안정성을 나타냅니다. 결함 발생 횟수와 피크 값 피처는 모델에게 순간적으로 튀는 값의 빈도를 알려줄 수 있습니다. 

### **미시적 시계열 rolling 파생변수**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `micro_sensor_rolling_std_mean_11` | 채널별 신호값의 이동 표준편차를 평균으로 집계, window=11 | `df[sensor_col].rolling(window=11, min_periods=1).std().mean()` |
| `micro_sensor_rolling_mean_std_11` | 채널별 신호값의 이동 평균을 표준편차로 집계, window=11 | `df[sensor_col].rolling(window=11, min_periods=1).mean().std()` |
| `micro_sensor_rolling_std_std_11` | 채널별 신호값의 이동 표준편차를 평균으로 집계, window=11 | `df[sensor_col].rolling(window=11, min_periods=1).std().std()` |
| `micro_sensor_rolling_std_mean_33` | 채널별 신호값의 이동 표준편차를 평균으로 집계, window=33 | `df[sensor_col].rolling(window=33, min_periods=1).std().mean()` |
| `micro_sensor_rolling_mean_std_33` | 채널별 신호값의 이동 평균을 표준편차로 집계, window=33 | `df[sensor_col].rolling(window=33, min_periods=1).mean().std()` |
| `micro_sensor_rolling_std_std_33` | 채널별 신호값의 이동 표준편차를 표준편차로 집계, window=33 | `df[sensor_col].rolling(window=33, min_periods=1).std().std()` |

*생성이유*: `t+1` 실험과 마찬가지로, EDA에서 확인된 원본 신호의 높은 변동성을 완화하고 신호의 국소적인(Local) 변화를 포착하기 위해 사용했습니다. 다만 실험을 통해 최적의 성능을 보인 window_size를 11과 33으로 변경하였습니다. 

### **미시적 시계열 ewm 파생변수**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `micro_sensor_ewm_11_mean` | 채널별 신호값의 지수가중평균의 최종값, span=11 | `df[sensor_col].ewm(span=11, min_periods=1).mean().iloc[-1]` |
| `micro_sensor_ewm_11_std` | 채널별 신호값을 지수가중표준편차의 최종값, span=11 | `df[sensor_col].ewm(span=11, min_periods=1).std().iloc[-1]` |
| `micro_sensor_ewm_33_mean` | 채널별 신호값을 지수가중평균의 최종값, span=33 | `df[sensor_col].ewm(span=33, min_periods=1).mean().iloc[-1]` |
| `micro_sensor_ewm_33_std` | 채널별 신호값을 지수가중표준편차의 최종값, span=33 | `df[sensor_col].ewm(span=33, min_periods=1).std().iloc[-1]` |

*생성이유*: `t+1` 실험과 동일하게, 최신 데이터에 더 큰 가중치를 부여하여 강철바의 최종 상태를 효과적으로 요약하기 위해 ewm 피처를 사용했습니다. rolling과 마찬가지로, span_size를 11과 33으로 변경하여 적용했습니다.

### **미시적 순열 엔트로피 파생변수**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
`sensor_col_perm_entropy` | 채널별 신호값의 순열 엔트로피(복잡도) | `ant.perm_entropy(df[sensor_col], normalize=True)` |

*생성이유*: amtropy 라이브러리의 순열 엔트로피(perm_entropy) 알고리즘을 사용하였으며, 순열 엔트로피는 시계열 데이터의 순서 관계를 기반으로 패턴의 무작위성 즉, 신호의 복잡성 또는 예측 불가능성을 측정하기 위해 사용합니다. 표준편차가 단순히 신호의 크기가 얼마나 변동하는지를 본다면, 순열 엔트로피는 신호의 패턴이 얼마나 무질서한지를 봅니다. 데이터의 값이 아닌, 값의 상대적인 순서에 주목하기 때문에 신호의 잡음에 강하다는 장점이 있습니다.
원래는 오리지널 entropy 알고리즘을 사용하려 하였으나 계산 시간이 너무 오래 걸려서 이보다 훨씬 가벼운 순열 엔트로피로 대체를 하게 되었습니다.

### **미시적 카츠 프랙탈 차원 파생변수**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `sensor_col_katz_fd` | 채널별 신호값의 카츠 프랙탈 차원(거칠기) | `ant.katz_fd(df[sensor_col])` |

*생성이유*: antropy 라이브러리의 카츠 프랙탈 차원(Katz Fractal Dimension) 알고리즘을 사용하였으며, 엔트로피와는 다른 관점에서, 신호의 기하학적인 거칠기를 측정하기 위해 사용합니다. 프랙탈 차원은 선형적인 통게량으로는 포착하기 힘든 신호의 복잡한 형태를 숫자로 나타냅니다. 

### **거시적 lag 파생변수**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `col_lag_1` | 채널별 기본 통계값에 shift(1) 적용 | `df.groupby(['LOT_ID'])[col].shift(1)` |
| `col_lag_3` | 채널별 기본 통계값에 shift(3) 적용 | `df.groupby(['LOT_ID'])[col].shift(3)` |
| `col_lag_5` | 채널별 기본 통계값에 shift(5) 적용 | `df.groupby(['LOT_ID'])[col].shift(5)` |
| `col_lag_9` | 채널별 기본 통계값에 shift(9) 적용 | `df.groupby(['LOT_ID'])[col].shift(9)` |

*생성이유*: lag는 시계열 데이터의 핵심 특성인 자기 상관관계를 모델에게 학습시키기 위한 피처입니다. 머신러닝 모델은 각 행을 독립적인 데이터로 간주하므로, `shift()`함수를 사용하여 과거 시점의 값을 새로운 피처로 추가함으로써 데이터의 시간 의존성을 주입합니다. lag가 적용되는 값은 이전에 산출했던 평균, 표준편차, 중위수 즉, 기본통계값에 적용합니다. 

### **거시적 rolling 파생변수**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `macro_col_rolling_mean_3` | 채널별 기본 통계값에 적용하는 이동 평균, window=3 | `df.groupby(['LOT_ID'])[col].rolling(window=3).mean()` |
| `macro_col_rolling_mean_11` | 채널별 기본 통계값에 적용하는 이동 평균, window=11 | `df.groupby(['LOT_ID'])[col].rolling(window=11).mean()` |
| `macro_col_rolling_std_3` | 채널별 기본 통계값에 적용하는 이동 표준편차, window=3 | `df.groupby(['LOT_ID'])[col].rolling(window=3).std()` |
| `macro_col_rolling_std_11` | 채널별 기본 통계값에 적용하는 이동 표준편차, window=11 | `df.groupby(['LOT_ID'])[col].rolling(window=11).std()` |

*생성이유*: EDA에서 확인을 했듯, 해당 데이터 프레임은 LOT 단위로 신호 패턴과 강종 등이 완전히 리셋이 되는 구조를 가집니다. 거시적 rolling 파생 변수는 LOT 내부에서 기본 통계 피처들의 시간에 따른 추세를 포착하기 위해 설계되었습니다. 이동평균 기법은 특정 크기window를 이동시키며 통계량을 계산하여, 단기적인 노이즈를 완화하고 데이터의 국소적인 흐름을 보여주는 가장 기본적인 방법입니다. window 크기를 3과 11로 설정한 것은, 모델에게 매우 짧은 기간의 즉각적인 추세(window=3)와조금 더 안정적인 중기적 추세(window=11)를 모두 제공하기 위함입니다. 

### **거시적 ewm 파생변수**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `macro_col_ewm_mean_3` | 채널별 기본 통계값에 적용하는 지수 이동 평균 | `df.groupby(['LOT_ID'])[col].ewm(span=3).mean()` |
| `macro_col_ewm_mean_11` | 채널별 기본 통계값에 적용하는 지수 이동 평균 | `df.groupby(['LOT_ID'])[col].ewm(span=11).mean()` |
| `macro_col_ewm_std_3` | 채널별 기본 통계값에 적용하는 지수 이동 표준편차 | `df.groupby(['LOT_ID'])[col].ewm(span=3).std()` |
| `macro_col_ewm_std_11` | 채널별 기본 통계값에 적용하는 지수 이동 표준편차 | `df.groupby(['LOT_ID'])[col].ewm(span=11).std()` |

*생성이유*: EDA에서 확인을 했듯, 해당 데이터 프레임은 LOT 단위로 신호 패턴과 강종 등이 완전히 리셋이 되는 구조를 가집니다. 거시적 ewm 파생 변수는 이러한 구조를 고려하여, 각 LOT 내부에서 기본 통계 피처들이 시간의 흐름에 따라 어떻게 변하는지 그 추세를 모델에게 알려주기 위해 설계가 되었습니다. ewm은 최신 강철바 데이터에 더 큰 가중치를 부여하기 때문에, 장비의 점진적인 상태 악화와 같은 변화 현상을 모델에 알려줄 수 있습니다.
span 값은 단기적이고 민감한 추세와 중기적이고 안정적인 추세를 제공하기 위해 설정되었습니다. 

### **거시적 LOT내부 컨텍스트 파생변수**
| Feature | Description | Calculation Method / Example |
| :--- | :--- | :--- |
| `bar_in_lot_sequence` | LOT 내에서 강철바의 상대적인 순서 | `df.groupby('LOT_ID').cumcount() + 1` |
| `normalized_sequence` | LOT내부 시간 정규화 | `(df['bar_in_lot_sequence'] - 1) / (df.groupby('LOT_ID')['bar_in_lot_sequence'].transform('max') - 1)` |
| `noise_delta_from_start` | LOT 시작점 대비 노이즈의 절대적 변화량 | `df[micro_sensor_rolling_std_mean_11] - df.groupby('LOT_ID')[noise_col].transform('first')` |
| `noise_ratio_from_start` | LOT 시작점 대비 노이즈의 상대적 변화율 | `df[micro_sensor_rolling_std_mean_11] / (start_noise + 1e-6)` |
| `seq_x_volatility` | 시간에 따라 가중된 변동성 | `df[bar_in_lot_sequence] * df[micro_sensor_rolling_std_mean_11]` |

*생성이유*: EDA에서 확인을 했듯, 해당 데이터 프레임은 LOT 단위로 신호 패턴과 강종 등이 완전히 리셋이 되는 구조를 가집니다. LOT 내부 컨텍스트 피처 그룹은 모델이 단순히 개별 강철바만 보는 것이 아닌, 해당 강철바가 소속된 LOT내에서 초반, 중반, 후반 중 어디쯤 위치해 있는가라는 맥락을 모델이 이해하도록 돕습니다. 또한 변동관련 피처들의 변동값들을 계산하기 위해서 `micro_sensor_rolling_std_mean_11`을 선택한 이유는 해당 컬림은 이동 표준편차의 평균으로 해당 강철바의 신호가 평균적으로 얼마나 흔들렸는가를 나타내는 즉, 신호의 국소적인 변동성과 불안정성을 가장 잘 나타내는 지표이기 때문이기 때문입니다.


3-4. 1차 모델링: 정적 피처 + 내부 신호 피처 + 미시적 rolling 파생변수

1번 타자 모델링
- 가설: 피처로는 미시적 시계열 rolling 파생변수가 추가가된 가운데, 문제의 난이도는 `t+1`에 비해 더 어려워진 상황에서 결정계수가 0에 가까운 양수를 띄게 될 것이다.
- 피처: 정적 피처 + 내부 신호 피처 + 미시적 rolling 파생변수
- 알고리즘: `RandomForestRegressor`
- 파라미터: `n_jobs=-1`(모든 CPU 코어 사용)
- 평가지표:
  - `r2_score: 0.2956`
  - `mae: 0.0112`
  - `mse: 0.0001`
- 결론: 너무 낮지도 높지도 않은 이상적인 결정계수 값과 비교적 높은 평균절대 오차가 나왔습니다. 이는 앞으로 추가되는 피처에 따라서 개선될 여지가 있는 굉장히 반가운 점수입니다.

2번 타자 모델링
- 가설: `RandomForestRegressor`에 비해 근소하게 높은 점수를 보일 것이다.
- 피처: 정적 피처 + 내부 신호 피처 + 미시적 rolling 파생변수
- 알고리즘: `XGBoostRegressor`
- 파라미터: `n_jobs=-1`(모든 CPU 코어 사용)
- 평가지표:
   - `r2_score: 0.3572`
   - `mae: 0.010`
   - `mse: 0.0001`
- 피처 중요도
<img width="677" height="551" alt="XGB3" src="https://github.com/user-attachments/assets/170a68e5-a3d5-47e2-96cd-00ce326d0f26" />

- SHAP value
<img width="551" height="677" alt="XGB4" src="https://github.com/user-attachments/assets/7d27c00d-ea31-4582-96f9-35e05c7c42f5" />

- 결론: `XGBoostRegressor`는 가설대로 `RandomForestRegressor`에 비해 근소한 성능 향상(`R²: 0.29 -> 0.35`)을 보였습니다. 모델의 성능에 큰 괴리가 없어 별도의 교차검증 없이 다음 단계로 진행했으며, `XGBoost` 모델의 피처 중요도와 `SHAP value`를 통해 다음과 같은 핵심 인사이트를 얻었습니다.
1. 피처 중요도 최상위에 `CH_A1_median`, `CH_A2_median`의 피처가 위치하여 있는데, A채널은 상단에 위치한 센서이며 B채널은 하단에 위치한 센서이고, 1과 2는 전방에 위치한 센서입니다. 전방에 위치한 센서는 검사시 강철바와 가장 먼저 조우하게 되는 센서들이며 해당 센서의 변수가 중요한 변수로 선택이 된 의미는 강철바가 센서에 처음 들어 오게 될 때 발생하는 신호값 그러니까 비유를 하자면 물줄기에 손가락을 넣는 순간 물이 튀는 것과 같이 모델이 검사가 시작될 때 발생하는 신호값을 가장 강한 예측 근거로 학습을 했을 가능성이 높다고 추측을 할 수 있습니다. 해당 내용은 모델이 단순히 강철바의 평균적인 신호값의 레벨 뿐 아니라, 검사 공정시 특정 단계에서 발생하는 이벤트에 가중을 두고 학습을 하는 것을 알 수 있습니다.
2. 새롭게 추가된 미시적 rolling 기반의 파생변수들 보다, 기본 통계 피처 중 중앙값의 중요도가 월등히 높게 나타났습니다. 중앙값은 다른 통계 값들인 평균과 표준편차에 비하여 이상치에 덜 민감한 특성을 갖는데, 이는 모델이 순간적으로 튀는 피크 값들 보다 강철바의 전반적으로 안정된 신호 레벨을 예측의 강한 근거로 사용하고 있음을 추측해 볼 수 있습니다.
3. `SHAP value` 분석 결과, `size`와 특정 강종의 피처들 또한 예측에 상당한 영향을 미치는 것으로 보입니다. 아직 `size`피처의 경우 긍정적인 영향인지 부정적인 영향을 주는지에 대해서는 정확하게 파악이 어렵지만, 이는 추가적으로 추가되는 피처에 따라 일어 나는 상호작용 여부에 따라 더 정확히 알 수 있을 것 같습니다.  




 




