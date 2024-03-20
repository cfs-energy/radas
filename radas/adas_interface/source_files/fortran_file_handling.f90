subroutine open_file(filename, file_unit)
    implicit none
    character(len=*), intent(in) :: filename
    integer, intent(in) :: file_unit
    open(unit=file_unit, file=filename)
end subroutine open_file

subroutine close_file(file_unit)
    implicit none
    integer, intent(in) :: file_unit
    close(unit=file_unit)
end subroutine close_file
