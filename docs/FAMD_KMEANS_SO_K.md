# Xác Định Số K Cho K-Means Trong Repo

Tài liệu này mô tả cách repo xác định số cụm `k` cho K-Means trong phần phân cụm trên không gian FAMD, repo đang dùng phương pháp nào, biểu đồ nào được tạo ra, và vì sao chọn cách đó.

## 1. Phần Code Liên Quan

Logic chọn `k` nằm chủ yếu trong:

```text
src/ml_models/famd.py
```

Các hàm chính:

- `FAMDAnalyzer.run_clustering()`: chạy K-Means và DBSCAN trên tọa độ FAMD.
- `FAMDAnalyzer.plot_cluster_projection()`: vẽ scatter plot các cụm.
- `FAMDAnalyzer.generate_clustering_report_html()`: tạo báo cáo HTML giải thích kết quả clustering.
- `FAMDAnalyzer.save_clustering_outputs()`: lưu JSON, biểu đồ và report HTML.

Điểm gọi từ pipeline:

```text
main.py -> run_famd_analysis() -> analyzer.save_clustering_outputs()
```

## 2. Repo Phân Cụm Trên Dữ Liệu Nào

Repo không chạy K-Means trực tiếp trên dữ liệu gốc.

Thay vào đó, quy trình là:

1. Chạy FAMD để biến đổi dữ liệu hỗn hợp numeric + categorical thành các trục tọa độ tiềm ẩn `F1`, `F2`, `F3`, ...
2. Lấy `n_dims` tọa độ FAMD đầu tiên qua `_famd_matrix()`.
3. Chuẩn hóa các tọa độ này bằng `StandardScaler`.
4. Chạy K-Means trên ma trận tọa độ FAMD đã chuẩn hóa.

Trong code:

```python
coords, component_cols = self._famd_matrix(n_dims=n_dims)
X = coords[component_cols].to_numpy(dtype=float)
X_scaled = StandardScaler().fit_transform(X)
```

Ý nghĩa:

- FAMD giúp nén dữ liệu hỗn hợp về một không gian số gọn hơn.
- K-Means hoạt động tốt hơn trên không gian liên tục, ít chiều hơn.
- Chuẩn hóa giúp các trục `F1`, `F2`, ... có cùng thang đo trước khi tính khoảng cách Euclidean.

## 3. Repo Chọn K Bằng Phương Pháp Nào

Repo dùng:

```text
Grid search trên k trong khoảng 2..10
```

Mặc định:

```python
k_range: tuple[int, int] = (2, 10)
```

Với mỗi giá trị `k`, repo:

1. Fit `KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")`
2. Tạo nhãn cụm
3. Tính các chỉ số đánh giá
4. Lưu toàn bộ candidate vào `kmeans_candidates`
5. Chọn mô hình có `silhouette` cao nhất làm `best_kmeans`

Đoạn chọn `k` cốt lõi:

```python
for k in range(k_min, k_max + 1):
    model = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
    labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    ...
    if best_kmeans is None or candidate["silhouette"] > best_kmeans["silhouette"]:
        best_kmeans = candidate
```

Kết luận ngắn:

- `k` được chọn theo `silhouette score` cao nhất.
- Đây là tiêu chí chính thức để chọn số cụm trong repo hiện tại.

### 3.1. Silhouette Là Gì Trong Repo Này

Silhouette là chỉ số đo xem từng điểm dữ liệu:

- gần với cụm của chính nó đến mức nào
- và tách khỏi cụm gần nhất kế bên ra sao

Với mỗi mẫu `i`, có thể hiểu:

- `a(i)`: khoảng cách trung bình từ mẫu `i` tới các điểm trong cùng cụm
- `b(i)`: khoảng cách trung bình nhỏ nhất từ mẫu `i` tới một cụm khác gần nhất

Công thức silhouette:

```text
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Giá trị của `s(i)` nằm trong khoảng `[-1, 1]`.

Cách đọc nhanh:

| Khoảng giá trị | Ý nghĩa |
|---|---|
| gần `1` | điểm nằm đúng cụm, cụm khá chặt và tách tốt |
| gần `0` | điểm nằm sát ranh giới giữa hai cụm |
| nhỏ hơn `0` | điểm có dấu hiệu bị gán vào cụm chưa phù hợp |

Trong dự án này, repo không tự đọc silhouette của từng điểm riêng lẻ để ra quyết định. Repo dùng **silhouette trung bình của toàn bộ tập dữ liệu** thông qua:

```python
silhouette = silhouette_score(X_scaled, labels)
```

Nghĩa là mỗi giá trị `k` sẽ có đúng một điểm số silhouette tổng hợp, và repo dùng điểm số này để so sánh giữa các phương án `k`.

### 3.2. Dự Án Chọn K Cụ Thể Ra Sao

Logic chọn `k` trong repo diễn ra theo đúng thứ tự sau:

1. Lấy các tọa độ FAMD đầu tiên từ `_famd_matrix()`.
2. Chuẩn hóa các tọa độ này bằng `StandardScaler`.
3. Đặt miền quét mặc định là `k = 2..10`.
4. Giới hạn lại cận trên theo dữ liệu thực tế:

```python
k_max = min(k_max, n_samples - 1)
```

Điểm này có nghĩa là repo không cố fit số cụm lớn hơn số mẫu có thể hỗ trợ.

5. Duyệt `k` theo thứ tự tăng dần:

```python
for k in range(k_min, k_max + 1):
```

6. Với mỗi `k`, repo fit `KMeans`, sinh nhãn cụm, rồi tính `silhouette`, `calinski_harabasz`, `davies_bouldin` và `inertia`.
7. Repo giữ lại ứng viên có `silhouette` lớn nhất làm `best_kmeans`.
8. Nếu hai giá trị `k` cho cùng một `silhouette`, repo sẽ giữ ứng viên đến trước trong vòng lặp, tức là **ưu tiên `k` nhỏ hơn**. Lý do là điều kiện trong code là:

```python
if best_kmeans is None or candidate["silhouette"] > best_kmeans["silhouette"]:
    best_kmeans = candidate
```

Code dùng dấu `>` chứ không dùng `>=`, nên trường hợp hòa điểm sẽ không ghi đè ứng viên cũ.

9. Sau cùng, repo ghi lại:

- `best_k`
- các metric của phương án thắng
- toàn bộ danh sách candidate trong `clustering["kmeans"]["candidates"]`

Vì vậy, cách chọn `K` của dự án là:

```text
Quét tuần tự k trên latent space FAMD đã chuẩn hóa, tính silhouette cho từng k, rồi lấy k có silhouette trung bình lớn nhất; nếu hòa thì giữ k nhỏ hơn.
```

## 4. Các Chỉ Số Repo Tính Cho Mỗi K

Mỗi candidate K-Means đều có các metric sau:

| Metric | Dùng để làm gì |
|---|---|
| `silhouette` | Tiêu chí chính để chọn `k` |
| `calinski_harabasz` | Chỉ số phụ để xem cụm có tách biệt tốt không; càng lớn thường càng tốt |
| `davies_bouldin` | Chỉ số phụ để xem cụm có chồng lấn không; càng nhỏ thường càng tốt |
| `inertia` | Tổng bình phương khoảng cách trong cụm; dùng để tham khảo độ chặt của cụm |

Repo lưu tất cả vào:

```text
clustering["kmeans"]["candidates"]
```

trong file:

```text
results/visualizations/famd_clustering_results.json
```

## 5. Tại Sao Repo Dùng Silhouette Để Chọn K

Repo ưu tiên `silhouette` vì đây là chỉ số cân bằng được hai yếu tố quan trọng:

- **Cohesion**: điểm trong cùng cụm có gần nhau không
- **Separation**: các cụm có tách xa nhau không

Điều này phù hợp với bài toán hiện tại vì clustering đang chạy trên tọa độ FAMD đã chuẩn hóa. Trong không gian đó, mục tiêu không chỉ là gom cụm thật chặt mà còn muốn các cụm có ý nghĩa phân tách tương đối rõ trên latent space.

Nói ngắn gọn theo đúng cách repo đang vận hành:

- `silhouette` là tiêu chí **ra quyết định**
- `Calinski-Harabasz`, `Davies-Bouldin` và `inertia` là tiêu chí **bổ sung để đọc kết quả**

Điều này cũng khớp với HTML report của repo, vì phần báo cáo clustering hiện ghi thẳng rằng K-Means chọn `k` theo silhouette cao nhất, sau đó mới hiển thị thêm các metric còn lại để diễn giải.

So với `inertia`:

- `inertia` gần như luôn giảm khi tăng `k`
- nên nếu chỉ nhìn `inertia`, ta cần thêm heuristic kiểu elbow/knee
- repo hiện **không cài thuật toán elbow detection riêng**

Vì vậy, chọn `k` theo `silhouette` là cách đơn giản, ổn định và có thể tự động hóa trực tiếp hơn trong code hiện tại.

## 6. Repo Có Dùng Elbow Method Không

Hiện tại:

```text
Không có elbow plot riêng và không có thuật toán phát hiện "điểm khuỷu" tự động.
```

Tuy nhiên repo vẫn lưu `inertia` cho từng `k`, nên:

- có thể dựng elbow plot sau này từ `kmeans.candidates`
- nhưng đó chưa phải output mặc định

Điều này rất quan trọng khi diễn giải:

- repo **có dữ liệu để kiểm tra theo hướng elbow**
- nhưng **quyết định cuối cùng hiện đang dựa trên silhouette**

## 7. Repo Sinh Ra Những Biểu Đồ Nào

### 7.1. Biểu đồ chính cho clustering

Repo hiện sinh các output trực tiếp liên quan đến K-Means:

| File | Ý nghĩa |
|---|---|
| `results/visualizations/famd_clusters_kmeans.html` | Scatter plot các điểm trên mặt phẳng FAMD, tô màu theo cụm K-Means |
| `results/visualizations/famd_clustering_report.html` | Báo cáo HTML tiếng Việt, nêu `k` được chọn và các metric chính |
| `results/visualizations/famd_clustering_results.json` | JSON đầy đủ chứa candidate metrics cho mọi `k` |

### 7.2. Biểu đồ FAMD hỗ trợ diễn giải cụm

Ngoài biểu đồ clustering trực tiếp, repo còn có các biểu đồ FAMD giúp hiểu tại sao cụm lại tách nhau:

| File | Ý nghĩa |
|---|---|
| `famd_variance_explained.html` | Xem từng component giải thích bao nhiêu phương sai |
| `famd_sample_projection_F1_F2.html`, `F2_F3`, ... | Xem phân bố điểm trên từng cặp trục FAMD |
| `famd_correlation_circle_F1_F2.html`, ... | Xem biến nào kéo mạnh theo từng trục |
| `famd_contributions_F1.html`, `F2.html`, ... | Xem biến đóng góp nhiều nhất cho từng component |

## 8. Tại Sao Dùng Các Biểu Đồ Này

### Scatter plot cụm K-Means

Repo dùng scatter plot trên mặt phẳng FAMD vì:

- dễ thấy cụm có tách khỏi nhau không
- dễ phát hiện cụm chồng lấn hoặc cụm kéo dài
- trực quan khi trình bày cho người không đọc code

Trong code, biểu đồ này dùng:

```python
px.scatter(..., color="cluster")
```

với nhãn trục có kèm tỷ lệ phương sai giải thích của từng component.

### Clustering report HTML

Repo dùng report HTML vì:

- cần diễn giải kết quả bằng tiếng Việt
- cần mô tả cụm theo `n_samples`, `share_pct`, `depression_rate`
- cần tóm tắt metric mà không bắt người xem mở JSON

### Variance explained + contribution + correlation circle

Những biểu đồ này không chọn `k` trực tiếp, nhưng rất quan trọng để trả lời:

- cụm đang tách nhau trên những trục nào
- các trục đó đại diện cho yếu tố gì
- biến nào chi phối cấu trúc cụm

Nói ngắn:

- scatter K-Means trả lời: `cụm có tách không`
- contribution/correlation circle trả lời: `vì sao cụm tách`

## 9. Repo Còn Dùng Phương Pháp Nào Ngoài K-Means

Repo còn chạy thêm:

```text
DBSCAN
```

Vai trò của DBSCAN trong repo:

- là phương pháp đối chiếu, không cần chọn `k`
- giúp kiểm tra xem dữ liệu có cấu trúc theo mật độ không
- giúp phát hiện noise/outlier

DBSCAN trong repo:

- tự quét `eps` theo các quantile của khoảng cách k-nearest-neighbor
- dùng `min_samples = max(5, 2 * số chiều)`
- chọn candidate tốt nhất theo:

```text
score = silhouette_non_noise - noise_fraction * 0.25
```

Điều này cho thấy repo không chỉ tin vào một thuật toán duy nhất, mà dùng K-Means làm phương pháp chính và DBSCAN làm đối chiếu hình thái cụm.

## 10. Cách Repo Diễn Giải Cụm Sau Khi Chọn K

Sau khi có nhãn cụm, repo không dừng ở việc trả ra cluster id.

Nó còn:

1. Tính mean của từng component trong cụm
2. Chọn 2 component nổi bật nhất theo trị tuyệt đối
3. Ghép với `top_contributions` của từng component
4. Tạo câu diễn giải tiếng Việt cho từng cụm

Ví dụ logic:

```text
F1 cao/thấp + F2 cao/thấp + component interpretation
```

Ngoài ra, nếu có cột `Depression`, repo chỉ dùng cột này để:

- tính `depression_rate` trong mỗi cụm
- hỗ trợ diễn giải sau khi clustering

Repo ghi rõ trong HTML report rằng:

```text
biến Depression chỉ dùng để diễn giải sau khi phân cụm
```

Điểm này đúng về mặt phương pháp vì tránh dùng nhãn mục tiêu để tạo cụm ngay từ đầu.

## 11. Hạn Chế Hiện Tại

Repo hiện có các giới hạn sau:

1. `k` được chọn theo silhouette duy nhất, chưa có voting/ranking đa tiêu chí.
2. Chưa có elbow plot HTML riêng.
3. Chưa có silhouette-vs-k line chart riêng.
4. Scatter plot mặc định chủ yếu nhìn trên cặp `F1-F2`, nên nếu cụm tách mạnh ở `F3-F4` thì cần mở thêm các biểu đồ FAMD khác để hiểu đầy đủ.

## 12. Kết Luận Ngắn

Tóm tắt đúng với repo hiện tại:

- Repo chọn số cụm `k` cho K-Means bằng cách quét `k = 2..10`.
- Dữ liệu phân cụm là các tọa độ FAMD đầu tiên đã được chuẩn hóa.
- Tiêu chí chọn `k` là `silhouette score` cao nhất.
- `Calinski-Harabasz`, `Davies-Bouldin` và `inertia` được lưu làm chỉ số bổ sung.
- Biểu đồ chính là scatter plot cụm trên mặt phẳng FAMD và báo cáo HTML giải thích cụm.
- Repo không dùng elbow plot làm tiêu chí quyết định chính ở phiên bản hiện tại.

## 13. File Output Nên Xem Theo Thứ Tự

Khi muốn giải thích clustering trước đám đông, nên mở theo thứ tự:

1. `results/visualizations/famd_variance_explained.html`
2. `results/visualizations/famd_clusters_kmeans.html`
3. `results/visualizations/famd_clustering_report.html`
4. `results/visualizations/famd_correlation_circle_F1_F2.html`
5. `results/visualizations/famd_contributions_F1.html` và `F2.html`

Thứ tự này giúp đi từ:

```text
không gian giảm chiều -> cụm nào đang thấy -> vì sao cụm đó hình thành
```
