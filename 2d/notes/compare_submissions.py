from src import save_rle_as_image

sub_16_filepath="C:\\Users\\ook\\Documents\\dev\\ashenvenus\\logs\\sub16.csv"
sub_12_filepath="C:\\Users\\ook\\Documents\\dev\\ashenvenus\\logs\\sub12.csv"
sub_11_filepath="C:\\Users\\ook\\Documents\\dev\\ashenvenus\\logs\\sub11.csv"
output_dir="C:\\Users\\ook\\Documents\\dev\\ashenvenus\\logs"

a_size = (2727, 6330)
b_size = (5454, 6330)

save_rle_as_image(sub_16_filepath, output_dir, 'a', a_size, '16')
save_rle_as_image(sub_16_filepath, output_dir, 'b', b_size, '16')
save_rle_as_image(sub_12_filepath, output_dir, 'a', a_size, '12')
save_rle_as_image(sub_12_filepath, output_dir, 'b', b_size, '12')
save_rle_as_image(sub_11_filepath, output_dir, 'a', a_size, '11')
save_rle_as_image(sub_11_filepath, output_dir, 'b', b_size, '11')