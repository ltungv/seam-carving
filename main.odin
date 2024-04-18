package main

import "core:fmt"
import "core:image"
import "core:image/png"
import "core:log"
import "core:math"
import "core:mem"
import "core:os"
import "core:slice"
import stbi "vendor:stb/image"

Matrix :: struct {
	width:    int,
	height:   int,
	stride:   int,
	elements: []f32,
}

Image :: struct {
	width:    int,
	height:   int,
	stride:   int,
	channels: int,
	pixels:   []f32,
}

main :: proc() {
	logger := log.create_console_logger(.Debug)
	defer log.destroy_console_logger(logger)

	context.logger = logger

	when ODIN_DEBUG {
		tracking_allocator_cleanup :: proc(track: ^mem.Tracking_Allocator) {
			if len(track.allocation_map) > 0 {
				log.errorf("=== %v allocations not freed: ===", len(track.allocation_map))
				for _, entry in track.allocation_map {
					log.errorf("- %v bytes @ %v", entry.size, entry.location)
				}
			}
			if len(track.bad_free_array) > 0 {
				log.errorf("=== %v incorrect frees: ===", len(track.bad_free_array))
				for entry in track.bad_free_array {
					log.errorf("- %p @ %v", entry.memory, entry.location)
				}
			}
			mem.tracking_allocator_destroy(track)
		}

		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		defer tracking_allocator_cleanup(&track)

		temp_track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&temp_track, context.temp_allocator)
		defer tracking_allocator_cleanup(&temp_track)

		context.allocator = mem.tracking_allocator(&track)
		context.temp_allocator = mem.tracking_allocator(&temp_track)
	}

	if !exec() do os.exit(1)
}

exec :: proc() -> bool {
	defer free_all(context.temp_allocator)

	img, ok := load_image("broadway_tower.png")
	if !ok do return false
	defer destroy_image(&img)

	grayscale := create_matrix(img.width, img.height)
	defer destroy_matrix(&grayscale)

	gradients := create_matrix(img.width, img.height)
	defer destroy_matrix(&gradients)

	energies := create_matrix(img.width, img.height)
	defer destroy_matrix(&energies)

	seam := make([]int, img.height)
	defer delete(seam)

	convert_image_to_grayscale(grayscale, img)
	for iter in 0 ..< 1000 {
		log.infof("Iteration %v", iter)
		calculate_image_gradients(gradients, grayscale)
		calculate_seam_energies(energies, gradients)
		find_minimum_seam(seam, energies)
		remove_seam_from_image(&img, seam)
		remove_seam_from_matrix(&grayscale, seam)
		gradients.width -= 1
		energies.width -= 1
		if (iter + 1) % 200 == 0 {
			name := fmt.aprintf("broadway_tower_carved_%v", iter + 1)
			defer delete(name)
			if !save_image_to_png(img, name) do return false
		}
	}

	return true
}

destroy_matrix :: proc(mat: ^Matrix) {
	if mat.elements != nil {
		delete(mat.elements)
		mat.elements = nil
	}
}

create_matrix :: proc(width, height: int) -> Matrix {
	mat := Matrix {
		width    = width,
		height   = height,
		stride   = width,
		elements = make([]f32, width * height),
	}
	return mat
}

get_matrix_element :: proc(mat: Matrix, idx: [2]int) -> f32 {
	element_idx := idx.y * mat.stride + idx.x
	return mat.elements[element_idx]
}

set_matrix_element :: proc(mat: Matrix, idx: [2]int, value: f32) {
	element_idx := idx.y * mat.stride + idx.x
	mat.elements[element_idx] = value
}

destroy_image :: proc(img: ^Image) {
	if img.pixels != nil {
		delete(img.pixels)
		img.pixels = nil
	}
}

create_image :: proc(width, height, channels: int) -> Image {
	img := Image {
		width    = width,
		height   = height,
		channels = channels,
		stride   = width,
		pixels   = make([]f32, width * height * channels),
	}
	return img
}

get_image_pixel :: proc(img: Image, idx: [2]int) -> []f32 {
	beg := idx.y * img.stride * img.channels + idx.x * img.channels
	end := beg + img.channels
	return img.pixels[beg:end]
}

set_image_pixel :: proc(img: Image, idx: [2]int, value: []f32) {
	assert(len(value) == img.channels)
	beg := idx.y * img.stride * img.channels + idx.x * img.channels
	end := beg + img.channels
	copy(img.pixels[beg:end], value[:img.channels])
}

load_image :: proc(path: string) -> (Image, bool) {
	normalize_pixels :: proc(normalized: []f32, pixels: [][$N]$T) {
		for pixel, i in pixels {
			for channel, j in normalize_pixel(pixel) {
				normalized[i * N + j] = channel
			}
		}
	}

	normalize_pixel :: proc "contextless" (pixel: [$N]$T) -> [N]f32 {
		normalized: [N]f32
		for x, i in pixel {
			normalized[i] = f32(x) / f32(max(T))
		}
		return normalized
	}

	img, err := image.load(path)
	if err != nil {
		log.errorf("Failed to load image: %v", err)
		return {}, false
	}
	defer image.destroy(img)

	if !image.is_valid_image(img) {
		log.error("Image format is invalid")
		return {}, false
	}

	if !image.alpha_drop_if_present(img, {.alpha_premultiply, .blend_background}) {
		log.error("Failed to drop alpha channel")
		return {}, false
	}

	result := create_image(img.width, img.height, img.channels)
	switch img.channels {
	case 1:
		switch img.depth {
		case 8:
			normalize_pixels(result.pixels, mem.slice_data_cast([]image.G_Pixel, img.pixels.buf[:]))
		case 16:
			normalize_pixels(result.pixels, mem.slice_data_cast([]image.G_Pixel_16, img.pixels.buf[:]))
		case:
			unreachable()
		}
	case 3:
		switch img.depth {
		case 8:
			normalize_pixels(result.pixels, mem.slice_data_cast([]image.RGB_Pixel, img.pixels.buf[:]))
		case 16:
			normalize_pixels(result.pixels, mem.slice_data_cast([]image.RGB_Pixel_16, img.pixels.buf[:]))
		case:
			unreachable()
		}
	case:
		unreachable()
	}
	return result, true
}

remove_seam_from_image :: proc(img: ^Image, seam: []int) {
	for col, row in seam {
		row_offset := row * img.stride * img.channels
		row_last_idx := row_offset + img.channels * img.width
		row_curr_idx := row_offset + img.channels * col
		copy(img.pixels[row_curr_idx:row_last_idx - img.channels], img.pixels[row_curr_idx + img.channels:row_last_idx])
	}
	img.width -= 1
}

remove_seam_from_matrix :: proc(mat: ^Matrix, seam: []int) {
	for col, row in seam {
		row_offset := row * mat.stride
		row_last_idx := row_offset + mat.width
		row_curr_idx := row_offset + col
		copy(mat.elements[row_curr_idx:row_last_idx - 1], mat.elements[row_curr_idx + 1:row_last_idx])
	}
	mat.width -= 1
}

convert_image_to_grayscale :: proc(dst: Matrix, src: Image) {
	convert_pixel_to_grayscale :: proc "contextless" (pixel: [3]f32) -> f32 {
		linearized: [3]f32
		for x, i in pixel {
			if x <= 0.04045 {
				linearized[i] = x / 12.92
			} else {
				linearized[i] = math.pow((x + 0.055) / 1.055, 2.4)
			}
		}
		luminance := 0.2126 * linearized.r + 0.7152 * linearized.g + 0.0722 * linearized.b
		lightness: f32
		if luminance <= 216.0 / 24389.0 {
			lightness = luminance * (24389.0 / 27.0)
		} else {
			lightness = math.pow(luminance, 1.0 / 3.0) * 116.0 - 16.0
		}
		return lightness / 100.0
	}

	assert(dst.width == src.width)
	assert(dst.height == src.height)
	switch src.channels {
	case 1:
		copy(dst.elements, src.pixels)
	case 3:
		for row in 0 ..< dst.height {
			for col in 0 ..< dst.width {
				pixel := get_image_pixel(src, {col, row})
				grayscale := convert_pixel_to_grayscale({pixel[0], pixel[1], pixel[2]})
				set_matrix_element(dst, {col, row}, grayscale)
			}
		}
	case:
		unreachable()
	}
}

calculate_image_gradients :: proc(gradients: Matrix, grayscale: Matrix) {
	assert(gradients.width == grayscale.width)
	assert(gradients.height == grayscale.height)
	hfilter := [?]f32{1, 2, 1, 0, 0, 0, -1, -2, -1}
	vfilter := [?]f32{1, 0, -1, 2, 0, -2, 1, 0, -1}
	for row in 0 ..< grayscale.height {
		for col in 0 ..< grayscale.width {
			hgradient: f32 = 0
			vgradient: f32 = 0
			for filter_row in 0 ..< 3 {
				for filter_col in 0 ..< 3 {
					pixel_row := row + filter_row - 1
					pixel_col := col + filter_col - 1
					if pixel_row < 0 || pixel_row >= grayscale.height || pixel_col < 0 || pixel_col >= grayscale.width {
						continue
					}
					gray_level := get_matrix_element(grayscale, {pixel_col, pixel_row})
					filter_idx := filter_row * 3 + filter_col
					hgradient += gray_level * hfilter[filter_idx]
					vgradient += gray_level * vfilter[filter_idx]
				}
			}
			gradient := math.sqrt(hgradient * hgradient + vgradient * vgradient)
			set_matrix_element(gradients, {col, row}, gradient)
		}
	}
}

calculate_seam_energies :: proc(energies: Matrix, gradients: Matrix) {
	assert(energies.width == gradients.width)
	assert(energies.height == gradients.height)
	for col in 0 ..< energies.width {
		gradient := get_matrix_element(gradients, {col, 0})
		set_matrix_element(energies, {col, 0}, gradient)
	}
	for row in 1 ..< energies.height {
		for col in 0 ..< energies.width {
			min_energy := max(f32)
			for prev_col in col - 1 ..= col + 1 {
				if prev_col < 0 || prev_col >= energies.width do continue
				energy := get_matrix_element(energies, {prev_col, row - 1})
				min_energy = min(min_energy, energy)
			}
			gradient := get_matrix_element(gradients, {col, row})
			set_matrix_element(energies, {col, row}, min_energy + gradient)
		}
	}
}

find_minimum_seam :: proc(seam: []int, energies: Matrix) -> []int {
	find_row_minimum :: proc(energies: Matrix, row: int, col_beg: int, col_end: int) -> int {
		min_col: int
		min_energy := max(f32)
		for col in col_beg ..= col_end {
			if col < 0 || col >= energies.width do continue
			energy := get_matrix_element(energies, {col, row})
			if energy < min_energy {
				min_col = col
				min_energy = energy
			}
		}
		return min_col
	}

	assert(len(seam) == energies.height)
	row := energies.height - 1
	col := find_row_minimum(energies, row, 0, energies.width - 1)
	seam[row] = col
	for row > 0 {
		row -= 1
		col = find_row_minimum(energies, row, col - 1, col + 1)
		seam[row] = col
	}
	return seam
}

save_matrix_to_png :: proc(mat: Matrix, name: string) -> bool {
	find_matrix_min_max :: proc(mat: Matrix) -> (f32, f32) {
		min_element := max(f32)
		max_element := min(f32)
		for row in 0 ..< mat.height {
			for col in 0 ..< mat.width {
				element := get_matrix_element(mat, {col, row})
				min_element = min(min_element, element)
				max_element = max(max_element, element)
			}
		}
		return min_element, max_element
	}

	buf := make([]u8, mat.width * mat.height)
	defer delete(buf)

	min_element, max_element := find_matrix_min_max(mat)
	for row in 0 ..< mat.height {
		for col in 0 ..< mat.width {
			idx := row * mat.width + col
			elem := get_matrix_element(mat, {col, row})
			norm := (elem - min_element) / (max_element - min_element)
			buf[idx] = u8(norm * 255)
		}
	}

	fname := fmt.caprintf("%s.png", name)
	defer delete(fname)

	write_err := stbi.write_png(fname, i32(mat.width), i32(mat.height), 1, slice.as_ptr(buf), i32(mat.width))
	return write_err != 0
}

save_image_to_png :: proc(img: Image, name: string) -> bool {
	buf := make([]u8, img.width * img.height * img.channels)
	defer delete(buf)

	for row in 0 ..< img.height {
		for col in 0 ..< img.width {
			idx := row * img.width * img.channels + col * img.channels
			for c, i in get_image_pixel(img, {col, row}) {
				buf[idx + i] = u8(c * 255)
			}
		}
	}

	fname := fmt.caprintf("%s.png", name)
	defer delete(fname)

	write_err := stbi.write_png(
		fname,
		i32(img.width),
		i32(img.height),
		i32(img.channels),
		slice.as_ptr(buf),
		i32(img.width) * i32(img.channels),
	)
	return write_err != 0
}
